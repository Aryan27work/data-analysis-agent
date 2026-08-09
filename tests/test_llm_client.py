import os
import sys
from unittest.mock import patch, MagicMock

import pytest
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_client import (
    call_with_retry,
    structured_call,
    plain_call,
    get_llm,
    LLMCallError,
    LLMConfigError,
)


class DummySchema(BaseModel):
    value: int


def test_get_llm_raises_config_error_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("utils.llm_client.get_api_key", return_value=None):
        with pytest.raises(LLMConfigError):
            get_llm()


def test_call_with_retry_succeeds_first_try():
    fn = MagicMock(return_value="ok")
    result = call_with_retry(fn)
    assert result == "ok"
    assert fn.call_count == 1


def test_call_with_retry_retries_then_succeeds():
    fn = MagicMock(side_effect=[ValueError("boom"), ValueError("boom"), "ok"])
    with patch("utils.llm_client.time.sleep"):  # skip real backoff delay
        result = call_with_retry(fn)
    assert result == "ok"
    assert fn.call_count == 3


def test_call_with_retry_exhausts_and_raises_llmcallerror():
    fn = MagicMock(side_effect=ValueError("persistent failure"))
    with patch("utils.llm_client.time.sleep"):
        with pytest.raises(LLMCallError) as exc_info:
            call_with_retry(fn)
    assert "persistent failure" in str(exc_info.value)
    assert fn.call_count == 3  # LLM_MAX_RETRIES default


def test_rate_limit_error_gets_longer_backoff():
    from utils.llm_client import _classify_backoff
    generic = _classify_backoff(ValueError("some generic error"), attempt=0)
    rate_limited = _classify_backoff(ValueError("429 rate limit exceeded"), attempt=0)
    assert rate_limited > generic


def test_structured_call_raises_on_malformed_output():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = None  # malformed/empty response
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("utils.llm_client.get_llm", return_value=mock_llm):
        with patch("utils.llm_client.time.sleep"):
            with pytest.raises(LLMCallError):
                structured_call("some prompt", DummySchema)


def test_structured_call_returns_valid_schema_instance():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = DummySchema(value=42)
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("utils.llm_client.get_llm", return_value=mock_llm):
        result = structured_call("some prompt", DummySchema)
    assert result.value == 42


def test_plain_call_raises_on_empty_response():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = ""
    mock_llm.invoke.return_value = mock_response

    with patch("utils.llm_client.get_llm", return_value=mock_llm):
        with patch("utils.llm_client.time.sleep"):
            with pytest.raises(LLMCallError):
                plain_call("some prompt")


def test_plain_call_returns_content_on_success():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Here is the report."
    mock_llm.invoke.return_value = mock_response

    with patch("utils.llm_client.get_llm", return_value=mock_llm):
        result = plain_call("some prompt")
    assert result == "Here is the report."


def test_simulated_timeout_is_retried_and_eventually_raises():
    fn = MagicMock(side_effect=TimeoutError("Deadline exceeded"))
    with patch("utils.llm_client.time.sleep"):
        with pytest.raises(LLMCallError):
            call_with_retry(fn)
    assert fn.call_count == 3
