"""Unit tests for residual-based demand anomaly detection."""

import numpy as np
import pandas as pd
import pytest
from src.anomaly.detector import ResidualAnomalyDetector


def test_residual_anomaly_detector_spike():
    # Construct series with baseline forecast and sudden 5x spike
    dates = pd.date_range("2024-01-01", periods=20)
    actuals = [10.0] * 19 + [65.0]  # Massive spike on day 20
    forecasts = [10.0] * 20

    df = pd.DataFrame({
        "item_id": ["FOODS_001"] * 20,
        "store_id": ["CA_1"] * 20,
        "date": dates,
        "actual": actuals,
        "forecast": forecasts,
    })

    detector = ResidualAnomalyDetector(z_threshold=3.0, rolling_window=7)
    annotated = detector.detect_anomalies(df)
    flagged = detector.get_flagged_anomalies(annotated)

    assert len(flagged) >= 1
    spike = flagged.iloc[0]
    assert spike["anomaly_type"] == "DEMAND_SPIKE"
    assert spike["severity"] == "HIGH"
    assert "replenishment" in spike["recommended_action"].lower()


def test_residual_anomaly_detector_stockout():
    # Construct series with consistent demand that abruptly collapses to 0 (stockout)
    dates = pd.date_range("2024-01-01", periods=15)
    actuals = [20.0] * 14 + [0.0]  # Zero sales stockout
    forecasts = [20.0] * 15

    df = pd.DataFrame({
        "item_id": ["FOODS_002"] * 15,
        "store_id": ["TX_1"] * 15,
        "date": dates,
        "actual": actuals,
        "forecast": forecasts,
    })

    detector = ResidualAnomalyDetector(z_threshold=3.0, rolling_window=7, min_expected_demand=5.0)
    annotated = detector.detect_anomalies(df)
    flagged = detector.get_flagged_anomalies(annotated)

    assert len(flagged) >= 1
    stockout = flagged[flagged["actual"] == 0.0].iloc[0]
    assert stockout["anomaly_type"] == "STOCKOUT_PLUNGE"
    assert "phantom inventory" in stockout["recommended_action"].lower()


def test_normal_noise_not_flagged():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=30)
    actuals = 15.0 + np.random.normal(0, 0.5, 30)
    forecasts = [15.0] * 30

    df = pd.DataFrame({
        "item_id": ["HOUSE_001"] * 30,
        "store_id": ["CA_2"] * 30,
        "date": dates,
        "actual": actuals,
        "forecast": forecasts,
    })

    detector = ResidualAnomalyDetector(z_threshold=3.0, rolling_window=7)
    annotated = detector.detect_anomalies(df)
    flagged = detector.get_flagged_anomalies(annotated)

    # Normal variations should not be flagged as anomalies
    assert len(flagged) == 0
