"""
Rolling-Origin (Walk-Forward) Chronological Backtester.
Simulates production forecasting over multiple chronological test origins,
comparing LightGBM against Seasonal-Naive and Prophet baselines without lookahead bias.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from src.features.engineer import prepare_feature_matrices
from src.models.baselines import ProphetBaseline, SeasonalNaiveBaseline
from src.models.lightgbm_model import RetailLightGBMModel
from src.utils.metrics import compute_all_metrics, wape


class RollingOriginBacktester:
    """
    Chronological Walk-Forward Backtester.
    Performs expanding-window evaluation across n_splits of length test_size_days.
    """

    def __init__(
        self,
        n_splits: int = 3,
        test_size_days: int = 28,
        step_size_days: int = 28,
        target_col: str = "sales",
    ):
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.step_size_days = step_size_days
        self.target_col = target_col
        self.fold_results: List[Dict[str, Any]] = []

    def run_backtest(
        self, df: pd.DataFrame, lgbm_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Execute walk-forward cross-validation across all origins.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        max_date = df["date"].max()

        print(f"[Backtester] Commencing {self.n_splits}-fold rolling-origin backtest...")
        print(f"[Backtester] Horizon: {self.test_size_days} days per fold. Step size: {self.step_size_days} days.")

        self.fold_results = []

        # Determine cutoff dates starting from most recent backwards
        cutoffs = []
        for i in range(self.n_splits):
            test_end = max_date - pd.Timedelta(days=i * self.step_size_days)
            test_start = test_end - pd.Timedelta(days=self.test_size_days - 1)
            cutoffs.append((test_start, test_end))

        cutoffs.reverse()  # Chronological order

        for fold_idx, (test_start, test_end) in enumerate(cutoffs, 1):
            print(f"\n--- Fold {fold_idx}/{self.n_splits}: Testing {test_start.date()} to {test_end.date()} ---")
            
            # Split
            train_val_df = df[df["date"] < test_start].copy()
            test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

            # Reserve last 14 days of train as internal validation for early stopping
            val_split_date = test_start - pd.Timedelta(days=14)
            train_df = train_val_df[train_val_df["date"] < val_split_date].copy()
            val_df = train_val_df[train_val_df["date"] >= val_split_date].copy()

            # 1. Evaluate Seasonal-Naive Baseline
            s_naive = SeasonalNaiveBaseline(seasonality=7)
            s_naive.fit(train_val_df, series_col="id", target_col=self.target_col)
            y_pred_naive = s_naive.predict(test_df, series_col="id")
            metrics_naive = compute_all_metrics(test_df[self.target_col].values, y_pred_naive)

            # 2. Evaluate Prophet Baseline
            prophet = ProphetBaseline()
            prophet.fit(train_val_df, series_col="id")
            y_pred_prophet = prophet.predict(test_df, series_col="id")
            metrics_prophet = compute_all_metrics(test_df[self.target_col].values, y_pred_prophet)

            # 3. Evaluate LightGBM
            X_train, y_train, X_val, y_val, X_test, y_test, encoder = prepare_feature_matrices(
                train_df, val_df, test_df, target_col=self.target_col
            )

            lgbm = RetailLightGBMModel(params=lgbm_params)
            lgbm.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=25)
            y_pred_lgbm = lgbm.predict(X_test)
            metrics_lgbm = compute_all_metrics(y_test.values, y_pred_lgbm)

            # Record
            for model_name, m_dict in [
                ("Seasonal_Naive", metrics_naive),
                ("Prophet", metrics_prophet),
                ("LightGBM", metrics_lgbm),
            ]:
                self.fold_results.append({
                    "fold": fold_idx,
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    "model": model_name,
                    **m_dict,
                })

            wape_reduction = (
                (metrics_naive["wape"] - metrics_lgbm["wape"]) / metrics_naive["wape"] * 100.0
            )
            print(f"Fold {fold_idx} WAPE -> Seasonal Naive: {metrics_naive['wape']:.1%} | Prophet: {metrics_prophet['wape']:.1%} | LightGBM: {metrics_lgbm['wape']:.1%}")
            print(f"Fold {fold_idx} Relative WAPE Reduction: {wape_reduction:.1f}%")

        res_df = pd.DataFrame(self.fold_results)
        return res_df

    def summarize_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate fold results into mean and std metrics per model."""
        summary = (
            results_df.groupby("model")[["wape", "rmse", "mae", "smape", "bias"]]
            .agg(["mean", "std"])
            .round(4)
        )
        return summary
