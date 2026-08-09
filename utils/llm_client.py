from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError

from utils.config import (
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    LLM_BASE_BACKOFF_SECONDS,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    get_api_key,
)
from utils.logging_config import get_logger


logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


RATE_LIMIT_MARKERS = (
    "rate limit",
    "resourceexhausted",
    "429",
    "quota",
    "too many requests",
)

TIMEOUT_MARKERS = (
    "timeout",
    "timed out",
    "deadline",
)


class LLMConfigError(RuntimeError):
    """Raised when the LLM cannot be configured."""


class LLMCallError(RuntimeError):
    """Raised after all LLM retries fail."""

    def __init__(
        self,
        message: str,
        original: Exception | None = None,
    ):
        super().__init__(message)
        self.original = original


def get_llm(
    temperature: float = GEMINI_TEMPERATURE,
):
    """
    Construct the shared Gemini chat model.

    Every LLM node uses this function so configuration is centralized.
    """

    api_key = get_api_key()

    if not api_key:
        raise LLMConfigError(
            "GEMINI_API_KEY is not configured. "
            "Add it to .env or Streamlit secrets."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=temperature,
        google_api_key=api_key,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=0,
    )


def _classify_backoff(
    exc: Exception,
    attempt: int,
) -> float:
    """
    Calculate exponential backoff.

    Rate-limit errors receive a longer delay.
    """

    message = str(exc).lower()

    delay = LLM_BASE_BACKOFF_SECONDS * (2**attempt)

    if any(marker in message for marker in RATE_LIMIT_MARKERS):
        delay *= 2

    return delay


def call_with_retry(
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Execute a callable with bounded exponential retries.
    """

    last_exc: Exception | None = None

    for attempt in range(LLM_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)

        except Exception as exc:
            last_exc = exc

            logger.warning(
                "llm_call_failed attempt=%d/%d type=%s",
                attempt + 1,
                LLM_MAX_RETRIES,
                type(exc).__name__,
            )

            if attempt >= LLM_MAX_RETRIES - 1:
                break

            delay = _classify_backoff(
                exc,
                attempt,
            )

            time.sleep(delay)

    raise LLMCallError(
        f"LLM call failed after {LLM_MAX_RETRIES} attempts: "
        f"{last_exc}",
        last_exc,
    )


def _content_to_text(content: Any) -> str:
    """
    Normalize LangChain/Gemini response content.

    Handles:
        str
        list[str]
        list[dict]
        other provider representations
    """

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                text = item.get("text")

                if text is not None:
                    parts.append(str(text))

        return "\n".join(parts).strip()

    return str(content).strip()


def _extract_json(text: str) -> Any:
    """
    Safely extract JSON from a model response.
    """

    text = text.strip()

    if not text:
        raise ValueError(
            "LLM returned an empty response."
        )

    # Markdown fenced JSON.
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced:
        text = fenced.group(1).strip()

    # Direct JSON.
    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # JSON object embedded inside explanation.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start : end + 1]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON: {exc}"
            ) from exc

    raise ValueError(
        "LLM response did not contain a valid JSON object."
    )


def _normalize_structured_result(
    result: Any,
    schema: Type[T],
) -> T:
    """
    Convert every supported LLM result shape into the requested
    Pydantic model.

    This function is intentionally strict about the final type.

    No list is allowed to leak into the graph where a Pydantic root
    object is expected.
    """

    if result is None:
        raise ValueError(
            "LLM returned an empty structured response."
        )

    if isinstance(result, schema):
        return result

    if isinstance(result, BaseModel):
        result = result.model_dump()

    if isinstance(result, str):
        result = _extract_json(result)

    if isinstance(result, dict):
        try:
            return schema.model_validate(result)

        except ValidationError as exc:
            raise ValueError(
                f"Structured output failed validation: {exc}"
            ) from exc

    raise ValueError(
        "Unsupported structured response type: "
        f"{type(result).__name__}. "
        f"Expected {schema.__name__}, dict, or JSON string."
    )


def _structured_call(
    llm: Any,
    prompt: str,
    schema: Type[T],
) -> T:
    """
    Current LangChain structured-output path.

    Crucially, we do NOT specify:

        method='json_mode'

    because that was the source of the previous unsupported-argument
    failure.
    """

    structured_llm = llm.with_structured_output(
        schema,
    )

    result = structured_llm.invoke(prompt)

    return _normalize_structured_result(
        result,
        schema,
    )


def _manual_json_call(
    llm: Any,
    prompt: str,
    schema: Type[T],
) -> T:
    """
    Defensive fallback.

    If native provider structured output fails because of a provider
    schema edge case, ask for JSON and validate it ourselves.
    """

    schema_json = json.dumps(
        schema.model_json_schema(),
        indent=2,
    )

    full_prompt = f"""
{prompt}

Return ONLY a single JSON object.

Do not return Markdown.
Do not use a code fence.
Do not add commentary.

The JSON must conform to this schema:

{schema_json}
""".strip()

    response = llm.invoke(full_prompt)

    content = getattr(
        response,
        "content",
        None,
    )

    text = _content_to_text(content)

    data = _extract_json(text)

    return _normalize_structured_result(
        data,
        schema,
    )


def structured_call(
    prompt: str,
    schema: Type[T],
    temperature: float = GEMINI_TEMPERATURE,
) -> T:
    """
    Execute a structured Gemini call.

    Strategy:

    1. Native LangChain structured output.
    2. If provider/schema conversion fails, retry using explicit JSON.
    3. Pydantic validates the final object.
    4. Only a validated Pydantic object leaves this function.
    """

    llm = get_llm(
        temperature=temperature,
    )

    def invoke() -> T:

        try:
            return _structured_call(
                llm,
                prompt,
                schema,
            )

        except Exception as native_exc:

            logger.warning(
                "native_structured_output_failed "
                "schema=%s error=%s",
                schema.__name__,
                native_exc,
            )

            return _manual_json_call(
                llm,
                prompt,
                schema,
            )

    return call_with_retry(invoke)


def plain_call(
    prompt: str,
    temperature: float = GEMINI_TEMPERATURE,
) -> str:
    """
    Execute a normal text-generation call.
    """

    llm = get_llm(
        temperature=temperature,
    )

    def invoke() -> str:

        response = llm.invoke(prompt)

        content = getattr(
            response,
            "content",
            None,
        )

        text = _content_to_text(content)

        if not text:
            raise ValueError(
                "LLM returned an empty response."
            )

        return text

    return call_with_retry(invoke)