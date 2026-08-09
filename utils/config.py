"""Central application configuration.

All configurable limits and LLM settings live here so that the rest of the
application can import them from one place.

Environment variables can override the defaults.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)

GEMINI_TEMPERATURE = float(
    os.getenv(
        "GEMINI_TEMPERATURE",
        "0.2",
    )
)

LLM_TIMEOUT_SECONDS = int(
    os.getenv(
        "LLM_TIMEOUT_SECONDS",
        "60",
    )
)

LLM_MAX_RETRIES = int(
    os.getenv(
        "LLM_MAX_RETRIES",
        "3",
    )
)

LLM_BASE_BACKOFF_SECONDS = float(
    os.getenv(
        "LLM_BASE_BACKOFF_SECONDS",
        "2",
    )
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
)


# ---------------------------------------------------------------------------
# Dataset safety / resource limits
# ---------------------------------------------------------------------------

MAX_FILE_MB = int(
    os.getenv(
        "MAX_FILE_MB",
        "50",
    )
)

MAX_ROWS_FOR_ANALYSIS = int(
    os.getenv(
        "MAX_ROWS_FOR_ANALYSIS",
        "100000",
    )
)

MAX_DETAILED_COLUMNS = int(
    os.getenv(
        "MAX_DETAILED_COLUMNS",
        "30",
    )
)


# ---------------------------------------------------------------------------
# LLM context budget
# ---------------------------------------------------------------------------

CONTEXT_TOKEN_CEILING = int(
    os.getenv(
        "CONTEXT_TOKEN_CEILING",
        "12000",
    )
)


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def get_api_key() -> str | None:
    """Read Gemini API key from environment first.

    If unavailable, fall back to Streamlit secrets when Streamlit is
    installed and secrets are configured.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key.strip()

    try:
        import streamlit as st

        value = st.secrets.get("GEMINI_API_KEY")

        if value:
            return str(value).strip()

    except Exception:
        pass

    return None