"""
Residual-Based Demand Anomaly Detection Module.
Detects supply chain anomalies by modeling forecast error distributions:
- Categorizes Demand Spikes (promotional surges, external viral shocks)
- Categorizes Demand Plunges / Stockouts (empty shelves, phantom inventory, logistics failures)
Computes Studentized Z-scores, severity ratings, and recommended supply chain actions.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class ResidualAnomalyDetector:
    """
    Supply Chain Residual Anomaly Detector.
    Evaluates residuals e_t = y_actual - y_forecast using dynamic rolling error statistics.
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        rolling_window: int = 14,
        min_expected_demand: float = 2.0,
        high_severity_threshold: float = 4.5,
        min_residual_magnitude: float = 2.0,
    ):
        self.z_threshold = z_threshold
        self.rolling_window = rolling_window
        self.min_expected_demand = min_expected_demand
        self.high_severity_threshold = high_severity_threshold
        self.min_residual_magnitude = min_residual_magnitude

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        actual_col: str = "actual",
        forecast_col: str = "forecast",
        series_cols: List[str] = ["item_id", "store_id"],
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Flag demand anomalies across time series using dynamic rolling residuals.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=series_cols + [date_col]).reset_index(drop=True)

        # 1. Residual calculation: e_t = actual - forecast
        df["residual"] = df[actual_col] - df[forecast_col]

        # 2. Dynamic rolling statistics per series
        grouped = df.groupby(series_cols)["residual"]
        # Rolling mean and std of residuals
        df["residual_mean"] = grouped.transform(
            lambda x: x.shift(1).rolling(self.rolling_window, min_periods=3).mean()
        ).fillna(0.0)
        
        df["residual_std"] = grouped.transform(
            lambda x: x.shift(1).rolling(self.rolling_window, min_periods=3).std()
        ).fillna(1.0)
        
        # Guard against zero/near-zero variance
        df["residual_std"] = np.where(df["residual_std"] < 0.2, 1.0, df["residual_std"])

        # 3. Dynamic Studentized Z-score
        df["z_score"] = (df["residual"] - df["residual_mean"]) / df["residual_std"]

        # 4. Anomaly Classification
        # Require statistical significance (|Z| >= threshold) AND practical supply chain magnitude
        is_spike = (df["z_score"] >= self.z_threshold) & (df["residual"] >= self.min_residual_magnitude)
        is_stockout = (
            (df["z_score"] <= -self.z_threshold) & (df["residual"].abs() >= self.min_residual_magnitude)
        ) | (
            (df[actual_col] == 0.0) & (df[forecast_col] >= self.min_expected_demand)
        )

        df["is_anomaly"] = is_spike | is_stockout

        # Assign Anomaly Type
        conditions = [
            is_spike,
            is_stockout,
        ]
        choices = [
            "DEMAND_SPIKE",
            "STOCKOUT_PLUNGE",
        ]
        df["anomaly_type"] = np.select(conditions, choices, default="NORMAL")

        # Assign Severity Rating
        abs_z = df["z_score"].abs()
        severity_conditions = [
            ~df["is_anomaly"],
            abs_z >= self.high_severity_threshold,
            abs_z >= self.z_threshold + 1.0,
        ]
        severity_choices = [
            "NONE",
            "HIGH",
            "MEDIUM",
        ]
        df["severity"] = np.select(severity_conditions, severity_choices, default="LOW")

        # Supply chain recommended actions
        action_map = {
            "DEMAND_SPIKE": "Trigger replenishment reorder; verify promotional campaign attribution.",
            "STOCKOUT_PLUNGE": "Execute phantom inventory audit; inspect shelf stock and shipment logs.",
            "NORMAL": "No action required.",
        }
        df["recommended_action"] = df["anomaly_type"].map(action_map)

        df["date"] = df[date_col].dt.strftime("%Y-%m-%d")
        return df

    def get_flagged_anomalies(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        """Filter to only flagged anomalous instances, sorted by severity and absolute Z-score."""
        anomalies = df_annotated[df_annotated["is_anomaly"]].copy()
        anomalies["abs_z"] = anomalies["z_score"].abs()
        return anomalies.sort_values(by="abs_z", ascending=False).drop(columns=["abs_z"])
