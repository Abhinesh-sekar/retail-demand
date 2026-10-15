"""
Data Loader & Dataset Validation Module.
Handles loading processed parquet features, chronological splitting, and schema integrity validation.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "item_id",
    "store_id",
    "date",
    "sales",
    "sell_price",
    "sales_lag_7",
    "rolling_mean_7",
]


def load_processed_data(parquet_path: str) -> pd.DataFrame:
    """Load featured retail time series dataset from parquet."""
    df = pd.read_parquet(parquet_path)
    validate_schema(df)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "item_id", "store_id"]).reset_index(drop=True)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure all required core columns and features exist in dataset."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def chronological_split(
    df: pd.DataFrame, test_days: int = 28, val_days: int = 28
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series chronologically to prevent temporal lookahead bias:
    - Train: Earliest up to T - (val_days + test_days)
    - Validation: T - (val_days + test_days) to T - test_days
    - Test: Last test_days
    """
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)

    train_df = df[df["date"] < val_start].copy()
    val_df = df[(df["date"] >= val_start) & (df["date"] < test_start)].copy()
    test_df = df[df["date"] >= test_start].copy()

    return train_df, val_df, test_df
