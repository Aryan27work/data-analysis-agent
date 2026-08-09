"""DataFrame-level content validation and safe cleanup."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from utils.config import MAX_ROWS_FOR_ANALYSIS
from utils.file_loader import FileLoadError
from utils.logging_config import get_logger


logger = get_logger(__name__)


def _dedupe_columns(
    df: pd.DataFrame,
    warnings: List[str],
) -> pd.DataFrame:

    if not df.columns.duplicated().any():
        return df

    seen: dict[str, int] = {}
    new_cols: list[str] = []
    dupes: list[str] = []

    for col in df.columns:

        col_str = str(col)

        if col_str in seen:

            seen[col_str] += 1

            new_name = (
                f"{col_str}__dup{seen[col_str]}"
            )

            new_cols.append(new_name)
            dupes.append(col_str)

        else:

            seen[col_str] = 0
            new_cols.append(col_str)

    df = df.copy()
    df.columns = new_cols

    warnings.append(
        "Duplicate column name(s) found and renamed "
        f"to keep them distinct: {sorted(set(dupes))}"
    )

    return df


def _replace_infinities(
    df: pd.DataFrame,
    warnings: List[str],
) -> pd.DataFrame:

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    if len(numeric_cols) == 0:
        return df

    inf_mask = np.isinf(
        df[numeric_cols]
    )

    inf_cols = [
        col
        for col in numeric_cols
        if inf_mask[col].any()
    ]

    if inf_cols:

        df = df.copy()

        df[numeric_cols] = df[
            numeric_cols
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        warnings.append(
            "Infinite numeric value(s) found in "
            f"column(s) {list(inf_cols)}; treated as "
            "missing (NaN) for analysis."
        )

    return df


def _flag_all_null_columns(
    df: pd.DataFrame,
    warnings: List[str],
) -> List[str]:

    all_null = [
        col
        for col in df.columns
        if df[col].isna().all()
    ]

    if all_null:
        warnings.append(
            "Column(s) entirely null/empty and excluded "
            f"from statistical analysis: {all_null}"
        )

    return all_null


def _cap_row_count(
    df: pd.DataFrame,
    warnings: List[str],
) -> pd.DataFrame:

    if len(df) <= MAX_ROWS_FOR_ANALYSIS:
        return df

    warnings.append(
        f"Dataset has {len(df):,} rows, exceeding the "
        f"{MAX_ROWS_FOR_ANALYSIS:,}-row analysis cap. "
        f"A random sample of {MAX_ROWS_FOR_ANALYSIS:,} "
        "rows was used instead, for resource safety."
    )

    return (
        df.sample(
            MAX_ROWS_FOR_ANALYSIS,
            random_state=0,
        )
        .reset_index(drop=True)
    )


def validate_and_clean(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Run DataFrame validation and safe repairs."""

    warnings: List[str] = []

    df = _dedupe_columns(
        df,
        warnings,
    )

    df = _replace_infinities(
        df,
        warnings,
    )

    df = _cap_row_count(
        df,
        warnings,
    )

    all_null_columns = _flag_all_null_columns(
        df,
        warnings,
    )

    usable_columns = [
        col
        for col in df.columns
        if col not in all_null_columns
    ]

    if not usable_columns:
        raise FileLoadError(
            "Every column in this dataset is entirely "
            "empty/null. There is nothing to analyze — "
            "please check the file and re-upload."
        )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    if duplicate_rows > 0:
        warnings.append(
            f"{duplicate_rows:,} fully duplicate row(s) "
            "detected (kept in the dataset, not removed "
            "automatically)."
        )

    logger.info(
        "validation_complete rows=%d cols=%d warnings=%d",
        df.shape[0],
        df.shape[1],
        len(warnings),
    )

    return (
        df,
        warnings,
        all_null_columns,
    )