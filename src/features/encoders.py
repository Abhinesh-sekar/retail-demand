"""
Categorical & Target Encoders for Retail Demand Forecasting.
Adheres strictly to the featurization ordering rule: encoders are fit on
training data ONLY, then applied to validation and test sets to eliminate data leakage.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class CategoricalLabelEncoder:
    """Ordinal / Label encoder for high-cardinality retail categorical columns."""

    def __init__(self, categorical_cols: List[str]):
        self.categorical_cols = categorical_cols
        self.mappings: Dict[str, Dict[str, int]] = {}

    def fit(self, df: pd.DataFrame) -> "CategoricalLabelEncoder":
        """Fit vocabulary mappings on training data only."""
        for col in self.categorical_cols:
            if col in df.columns:
                unique_vals = sorted(df[col].dropna().unique())
                # 0 is reserved for unseen / unknown categories
                self.mappings[col] = {val: idx + 1 for idx, val in enumerate(unique_vals)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply mappings to dataframe without modifying original."""
        df_out = df.copy()
        for col in self.categorical_cols:
            if col in self.mappings and col in df_out.columns:
                mapping = self.mappings[col]
                df_out[col] = df_out[col].map(mapping).fillna(0).astype(int)
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class OutOfFoldTargetEncoder:
    """
    Out-Of-Fold (OOF) Target Mean Encoder with additive smoothing.
    Fits target means on training folds, applying global prior smoothing:
    smooth_mean = (count * mean + m * prior) / (count + m)
    """

    def __init__(self, cols: List[str], target_col: str = "sales", smoothing: float = 10.0):
        self.cols = cols
        self.target_col = target_col
        self.smoothing = smoothing
        self.global_mean: float = 0.0
        self.encoding_maps: Dict[str, Dict[Any, float]] = {}

    def fit(self, df: pd.DataFrame) -> "OutOfFoldTargetEncoder":
        self.global_mean = float(df[self.target_col].mean())
        for col in self.cols:
            if col in df.columns:
                stats = df.groupby(col)[self.target_col].agg(["count", "mean"])
                smooth = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean) / (
                    stats["count"] + self.smoothing
                )
                self.encoding_maps[col] = smooth.to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in self.cols:
            if col in self.encoding_maps and col in df_out.columns:
                new_col = f"{col}_target_enc"
                df_out[new_col] = df_out[col].map(self.encoding_maps[col]).fillna(self.global_mean)
        return df_out
