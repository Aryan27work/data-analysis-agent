"""Defensive CSV/Excel loading.

Parsing errors are converted into safe FileLoadError messages instead of
raw tracebacks being exposed to the user.
"""

from __future__ import annotations

import io
from typing import Union

import pandas as pd

from utils.config import MAX_FILE_MB
from utils.logging_config import get_logger


logger = get_logger(__name__)


class FileLoadError(Exception):
    """Safe, user-facing error raised when a file cannot be loaded."""


SUPPORTED_EXTENSIONS = (
    ".csv",
    ".xlsx",
    ".xls",
)


def check_file_size(
    size_bytes: int,
    max_mb: int = MAX_FILE_MB,
) -> None:

    if size_bytes > max_mb * 1024 * 1024:
        raise FileLoadError(
            f"File exceeds the {max_mb}MB limit. "
            "Please upload a smaller sample of the dataset."
        )


def load_dataframe(
    file: Union[str, io.BytesIO],
    filename: str,
) -> pd.DataFrame:
    """Load CSV/XLSX/XLS into a pandas DataFrame."""

    lower = filename.lower()

    if not lower.endswith(
        SUPPORTED_EXTENSIONS
    ):
        raise FileLoadError(
            f"Unsupported file type for '{filename}'. "
            f"Please upload a "
            f"{', '.join(SUPPORTED_EXTENSIONS)} file."
        )

    # Check size where possible.
    try:
        if isinstance(file, str):
            import os

            check_file_size(
                os.path.getsize(file)
            )

        elif hasattr(file, "getbuffer"):
            check_file_size(
                file.getbuffer().nbytes
            )

        elif hasattr(file, "seek") and hasattr(file, "tell"):
            current = file.tell()

            file.seek(
                0,
                2,
            )

            size = file.tell()

            file.seek(
                current
            )

            check_file_size(size)

    except FileLoadError:
        raise

    except Exception:
        # Size checking should never prevent a legitimate load.
        pass

    try:

        if lower.endswith(".csv"):
            df = pd.read_csv(file)

        else:
            df = pd.read_excel(file)

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:

        raise FileLoadError(
            f"'{filename}' could not be parsed — "
            "the file appears to be empty or corrupted. "
            f"({exc.__class__.__name__})"
        ) from exc

    except UnicodeDecodeError as exc:

        raise FileLoadError(
            f"'{filename}' could not be decoded as text. "
            "If this is an Excel file, make sure it has a "
            ".xlsx/.xls extension."
        ) from exc

    except Exception as exc:

        logger.warning(
            "file_load_failed filename=%s error_type=%s",
            filename,
            exc.__class__.__name__,
        )

        raise FileLoadError(
            f"Could not parse '{filename}' as a valid "
            f"CSV/Excel file. ({exc.__class__.__name__})"
        ) from exc

    if df.shape[0] == 0:
        raise FileLoadError(
            "The uploaded file has no rows. "
            "Please upload a dataset with data."
        )

    if df.shape[1] == 0:
        raise FileLoadError(
            "The uploaded file has no columns. "
            "Please check the file and re-upload."
        )

    return df