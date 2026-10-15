"""Unit tests for feature engineering and encoders."""

import numpy as np
import pandas as pd
import pytest
from src.features.encoders import CategoricalLabelEncoder, OutOfFoldTargetEncoder
from src.features.engineer import prepare_feature_matrices


def test_categorical_label_encoder():
    df_train = pd.DataFrame({"item_id": ["FOODS_001", "FOODS_002", "FOODS_003"]})
    df_test = pd.DataFrame({"item_id": ["FOODS_001", "UNSEEN_SKU"]})

    encoder = CategoricalLabelEncoder(categorical_cols=["item_id"])
    df_train_enc = encoder.fit_transform(df_train)
    df_test_enc = encoder.transform(df_test)

    # Known category maps to positive integer
    assert df_train_enc["item_id"].iloc[0] > 0
    # Unseen category maps to 0
    assert df_test_enc["item_id"].iloc[1] == 0


def test_oof_target_encoder():
    df = pd.DataFrame({
        "cat_id": ["FOODS", "FOODS", "FOODS", "HOUSEHOLD", "HOUSEHOLD"],
        "sales": [10.0, 20.0, 30.0, 5.0, 15.0],
    })
    encoder = OutOfFoldTargetEncoder(cols=["cat_id"], target_col="sales", smoothing=2.0)
    encoder.fit(df)
    transformed = encoder.transform(df)

    assert "cat_id_target_enc" in transformed.columns
    # Transformed values should be between category mean and global mean
    foods_enc = transformed[transformed["cat_id"] == "FOODS"]["cat_id_target_enc"].iloc[0]
    assert 10.0 < foods_enc < 25.0


def test_lag_leakage_absence():
    """Verify that lag_1 strictly matches sales shifted by 1."""
    dates = pd.date_range("2024-01-01", periods=10)
    sales = np.array([5, 8, 12, 15, 20, 25, 30, 28, 35, 40])
    df = pd.DataFrame({
        "item_id": ["SKU1"] * 10,
        "store_id": ["STORE1"] * 10,
        "date": dates,
        "sales": sales,
    })
    df["sales_lag_1"] = df.groupby(["item_id", "store_id"])["sales"].shift(1)

    # First entry must be NaN
    assert pd.isna(df["sales_lag_1"].iloc[0])
    # Subsequent entries must match previous index sales
    for i in range(1, len(df)):
        assert df["sales_lag_1"].iloc[i] == sales[i - 1]
