"""Unit tests for models and evaluation metrics."""

import numpy as np
import pandas as pd
import pytest
from src.models.baselines import ProphetBaseline, SeasonalNaiveBaseline
from src.models.lightgbm_model import RetailLightGBMModel
from src.utils.metrics import compute_all_metrics, forecast_bias, mae, rmse, wape


def test_metric_calculations():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 35.0])

    # sum(|err|) = 2 + 2 + 3 + 5 = 12. sum(y_true) = 100. WAPE = 12/100 = 0.12
    assert pytest.approx(wape(y_true, y_pred), 1e-4) == 0.12
    assert pytest.approx(mae(y_true, y_pred), 1e-4) == 3.0
    # sum(pred - true) = 2 - 2 + 3 - 5 = -2. Bias = -2/100 = -0.02
    assert pytest.approx(forecast_bias(y_true, y_pred), 1e-4) == -0.02

    metrics = compute_all_metrics(y_true, y_pred)
    assert "wape" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics


def test_seasonal_naive_baseline():
    df_train = pd.DataFrame({
        "id": ["SKU1_CA1"] * 14,
        "sales": [10, 12, 14, 16, 18, 20, 22, 11, 13, 15, 17, 19, 21, 23],
    })
    df_test = pd.DataFrame({
        "id": ["SKU1_CA1"] * 2,
        "sales_lag_7": [15, 17],
    })

    model = SeasonalNaiveBaseline(seasonality=7)
    model.fit(df_train, series_col="id")
    preds = model.predict(df_test, series_col="id")

    assert len(preds) == 2
    assert preds[0] == 15.0
    assert preds[1] == 17.0


def test_lightgbm_fit_and_predict(tmp_path):
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "sales_lag_1": np.random.uniform(5, 25, n),
        "sales_lag_7": np.random.uniform(5, 25, n),
        "sell_price": np.random.uniform(2, 10, n),
        "wday": np.random.randint(0, 7, n),
    })
    y = pd.Series(X["sales_lag_1"] * 0.5 + X["sales_lag_7"] * 0.5 + np.random.normal(0, 1, n))

    model = RetailLightGBMModel(params={"n_estimators": 20, "learning_rate": 0.1, "verbose": -1})
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == n
    assert (preds >= 0).all()  # Non-negative constraint

    imp = model.get_feature_importance()
    assert len(imp) == 4

    # Test serialization
    save_file = str(tmp_path / "model.joblib")
    model.save(save_file)
    loaded = RetailLightGBMModel.load(save_file)
    loaded_preds = loaded.predict(X)
    np.testing.assert_allclose(preds, loaded_preds)
