import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.file_loader import load_dataframe, check_file_size, FileLoadError
from utils.validation import validate_and_clean


# --- file_loader ---------------------------------------------------------

def test_valid_csv_loads():
    buf = io.BytesIO(b"a,b\n1,2\n3,4\n")
    df = load_dataframe(buf, "data.csv")
    assert df.shape == (2, 2)


def test_valid_excel_loads():
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    buf = io.BytesIO()
    df_in.to_excel(buf, index=False)
    buf.seek(0)
    df = load_dataframe(buf, "data.xlsx")
    assert df.shape == (2, 2)


def test_empty_file_rejected():
    buf = io.BytesIO(b"")
    with pytest.raises(FileLoadError):
        load_dataframe(buf, "empty.csv")


def test_corrupted_excel_rejected():
    buf = io.BytesIO(b"this is not a real xlsx file, just garbage bytes")
    with pytest.raises(FileLoadError):
        load_dataframe(buf, "corrupt.xlsx")


def test_unsupported_extension_rejected():
    buf = io.BytesIO(b"whatever")
    with pytest.raises(FileLoadError):
        load_dataframe(buf, "data.txt")


def test_zero_row_csv_rejected():
    buf = io.BytesIO(b"a,b,c\n")
    with pytest.raises(FileLoadError):
        load_dataframe(buf, "headeronly.csv")


def test_file_size_limit_enforced():
    with pytest.raises(FileLoadError):
        check_file_size(size_bytes=200 * 1024 * 1024, max_mb=50)


def test_file_size_within_limit_ok():
    check_file_size(size_bytes=1024, max_mb=50)  # should not raise


# --- validation (dataframe-level) -----------------------------------------

def test_duplicate_columns_are_deduplicated():
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
    cleaned, warnings, _ = validate_and_clean(df)
    assert len(set(cleaned.columns)) == 3
    assert any("Duplicate column" in w for w in warnings)


def test_infinite_values_replaced_with_nan():
    df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 4.0]})
    cleaned, warnings, _ = validate_and_clean(df)
    assert cleaned["a"].isin([np.inf, -np.inf]).sum() == 0
    assert any("Infinite" in w for w in warnings)


def test_all_null_column_flagged_not_dropped():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
    cleaned, warnings, all_null = validate_and_clean(df)
    assert "b" in cleaned.columns  # not dropped, still present
    assert all_null == ["b"]
    assert any("entirely null" in w for w in warnings)


def test_all_columns_null_raises():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(FileLoadError):
        validate_and_clean(df)


def test_mixed_datatypes_do_not_crash():
    df = pd.DataFrame({"a": [1, "two", 3.0, None]})
    cleaned, warnings, _ = validate_and_clean(df)
    assert cleaned.shape[0] == 4


def test_duplicate_rows_flagged():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [5, 5, 6]})
    cleaned, warnings, _ = validate_and_clean(df)
    assert any("duplicate row" in w for w in warnings)


def test_large_row_count_is_sampled_down():
    df = pd.DataFrame({"a": range(300_000)})
    cleaned, warnings, _ = validate_and_clean(df)
    assert cleaned.shape[0] <= 200_000
    assert any("row" in w and "cap" in w for w in warnings)
