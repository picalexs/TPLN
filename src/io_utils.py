"""I/O utilities for data loading and saving."""

import os
import pandas as pd
from .paths import CLEAN_PARQUET


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

    df = pd.read_parquet(CLEAN_PARQUET, columns=columns)
    if nrows is not None:
        df = df.head(nrows).copy()

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
    """Backward-compatible alias for loading the cleaned dataset."""
    return load_clean_data(nrows=nrows)
