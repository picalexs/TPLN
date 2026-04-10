"""I/O utilities for data loading and saving."""

import os
import pandas as pd
from .paths import CLEAN_CSV


def load_clean_csv(nrows=None):
    """Load the cleaned CSV with optional row limit.

    Assumes data has already been cleaned by DataCuration.py.
    Fills NaN values in required text columns and validates the expected schema.
    """
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(
            f"Cleaned CSV not found at {CLEAN_CSV}.\n"
            "Run DataCuration.py first."
        )

    df = pd.read_csv(CLEAN_CSV, nrows=nrows)

    required_cols = ["title", "document", "short_document", "document_nostop", "topics"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Cleaned CSV is missing required columns: "
            f"{', '.join(missing_cols)}. Re-run DataCuration.py."
        )

    for col in required_cols:
        df[col] = df[col].fillna("").astype(str)

    return df
