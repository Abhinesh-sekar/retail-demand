"""
Feature Engineering & Dataset Preparation.
Selects and formats model feature matrices, handling missing values and column definitions.
"""

from typing import List, Tuple
import pandas as pd
from src.features.encoders import CategoricalLabelEncoder


FEATURE_COLUMNS = [
    # Categorical
    "item_id",
    "store_id",
    "dept_id",
    "cat_id",
    "state_id",
    # Price
    "sell_price",
    "price_discount_ratio",
    # Calendar
    "wday",
    "month",
    "dayofmonth",
    "is_weekend",
    "sin_dow",
    "cos_dow",
    "sin_month",
    "cos_month",
    "has_event",
    "snap_CA",
    "snap_TX",
    # Lags
    "sales_lag_1",
    "sales_lag_2",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_21",
    "sales_lag_28",
    # Rolling stats
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_28",
    "rolling_std_28",
    "rolling_mean_60",
    "rolling_std_60",
]


def prepare_feature_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] = None,
    target_col: str = "sales",
    categorical_cols: List[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, CategoricalLabelEncoder]:
    """
    Encode categoricals and prepare X, y matrices for train, validation, and test.
    Fits encoder strictly on training data.
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]

    if categorical_cols is None:
        categorical_cols = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]

    # Drop burn-in rows in train where lags/rolling windows are NaN (first 60 days)
    clean_train = train_df.dropna(subset=feature_cols + [target_col]).copy()
    clean_val = val_df.dropna(subset=feature_cols + [target_col]).copy()
    clean_test = test_df.dropna(subset=feature_cols + [target_col]).copy()

    # Fit encoder on train ONLY
    encoder = CategoricalLabelEncoder(categorical_cols=categorical_cols)
    clean_train = encoder.fit_transform(clean_train)
    clean_val = encoder.transform(clean_val)
    clean_test = encoder.transform(clean_test)

    X_train = clean_train[feature_cols]
    y_train = clean_train[target_col]

    X_val = clean_val[feature_cols]
    y_val = clean_val[target_col]

    X_test = clean_test[feature_cols]
    y_test = clean_test[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test, encoder
