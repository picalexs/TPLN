"""I/O utilities for data loading and saving."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .paths import CLEAN_PARQUET


def _read_parquet_in_batches(
    parquet_path: Path,
    columns: list[str] | None = None,
    nrows: int | None = None,
    batch_size: int = 50_000,
) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(str(parquet_path))
    frames: list[pd.DataFrame] = []
    rows_read = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        if nrows is not None:
            remaining = nrows - rows_read
            if remaining <= 0:
                break
            if len(frame) > remaining:
                frame = frame.iloc[:remaining].copy()
            rows_read += len(frame)
        frames.append(frame)
        if nrows is not None and rows_read >= nrows:
            break

    if not frames:
        return pd.DataFrame(columns=columns or [])

    return pd.concat(frames, ignore_index=True)


def load_clean_data(nrows=None, columns=None):
    """Load the cleaned dataset with optional row limit and column subset.

    Assumes data has already been cleaned by scripts/DataCuration.py.
    Fills NaN values in required text columns and validates the expected schema.
    """
    if not os.path.exists(CLEAN_PARQUET):
        raise FileNotFoundError(
            f"Cleaned dataset not found at {CLEAN_PARQUET}.\n"
            "Run scripts/DataCuration.py first."
        )

    df = _read_parquet_in_batches(Path(CLEAN_PARQUET), columns=columns, nrows=nrows)

    required_cols = ["title", "document", "short_document", "document_nostop", "topics"]
    required_for_this_load = (
        required_cols
        if columns is None
        else [col for col in required_cols if col in columns]
    )
    missing_cols = [col for col in required_for_this_load if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Cleaned parquet is missing required columns: "
            f"{', '.join(missing_cols)}. Re-run scripts/DataCuration.py."
        )

    for col in required_for_this_load:
        df[col] = df[col].fillna("").astype(str)

    return df


def load_clean_csv(nrows=None):
    """Deprecated alias for load_clean_data(). Will be removed in a future version."""
    import warnings
    warnings.warn(
        "load_clean_csv() is deprecated. Use load_clean_data() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_clean_data(nrows=nrows)
