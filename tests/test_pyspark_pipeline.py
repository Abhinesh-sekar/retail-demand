"""Unit tests for PySpark / Vectorized ETL pipeline."""

from pathlib import Path
import pandas as pd
import pytest
from src.data.pyspark_pipeline import run_vectorized_feature_pipeline


def test_vectorized_feature_pipeline(tmp_path):
    # Setup miniature test tables
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_file = str(tmp_path / "processed" / "test_featured.parquet")

    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    sales_df = pd.DataFrame({
        "item_id": ["SKU1"] * 4,
        "store_id": ["S1"] * 4,
        "date": dates,
        "sales": [10.0, 15.0, 20.0, 25.0],
    })
    cal_df = pd.DataFrame({
        "date": dates,
        "wm_yr_wk": [202401] * 4,
        "wday": [0, 1, 2, 3],
        "event_name_1": [None, None, "Promo", None],
    })
    prices_df = pd.DataFrame({
        "store_id": ["S1"],
        "item_id": ["SKU1"],
        "wm_yr_wk": [202401],
        "sell_price": [9.99],
    })

    sales_path = str(raw_dir / "sales.parquet")
    cal_path = str(raw_dir / "cal.parquet")
    prices_path = str(raw_dir / "prices.parquet")

    sales_df.to_parquet(sales_path, index=False)
    cal_df.to_parquet(cal_path, index=False)
    prices_df.to_parquet(prices_path, index=False)

    df_out = run_vectorized_feature_pipeline(
        sales_path=sales_path,
        calendar_path=cal_path,
        prices_path=prices_path,
        output_path=out_file,
        lags=[1, 2],
        rolling_windows=[2],
    )

    assert Path(out_file).exists()
    assert "sales_lag_1" in df_out.columns
    assert "rolling_mean_2" in df_out.columns
    assert "sin_dow" in df_out.columns
    assert "price_discount_ratio" in df_out.columns
    assert len(df_out) == 4
