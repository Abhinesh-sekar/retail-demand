"""
Evaluation metrics for retail supply chain demand forecasting.
Features WAPE (Weighted Absolute Percentage Error), RMSE, MAE, SMAPE, and Forecast Bias.
"""

from typing import Dict
import numpy as np


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Weighted Absolute Percentage Error (WAPE).
    WAPE = sum(|y_true - y_pred|) / sum(y_true)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0 if np.sum(np.abs(y_pred)) == 0 else float("inf")
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Symmetric Mean Absolute Percentage Error (SMAPE) in percentage (0-100%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(np.abs(y_pred - y_true) / denominator) * 100.0)


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Forecast Tracking Bias.
    Positive indicates over-forecasting, negative indicates under-forecasting.
    Bias = sum(y_pred - y_true) / sum(y_true)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float(np.sum(y_pred - y_true) / total_actual)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive dictionary of time series forecast metrics."""
    return {
        "wape": round(wape(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "smape": round(smape(y_true, y_pred), 2),
        "bias": round(forecast_bias(y_true, y_pred), 4),
    }
