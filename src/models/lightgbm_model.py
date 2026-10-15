"""
LightGBM Retail Demand Forecasting Model.
Employs Tweedie loss for zero-inflated demand distributions,
early stopping on chronological validation splits, and feature importance analysis.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


DEFAULT_LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.15,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 400,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}


class RetailLightGBMModel:
    """Wrapper for LightGBM Regressor tailored to retail time series forecasting."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = DEFAULT_LGBM_PARAMS.copy()
        if params:
            self.params.update(params)
        
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: List[str] = []
        self.best_iteration: Optional[int] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
        early_stopping_rounds: int = 30,
    ) -> "RetailLightGBMModel":
        """
        Fit LightGBM model with optional early stopping on validation split.
        """
        self.feature_names = list(X_train.columns)
        fit_params = self.params.copy()
        callbacks = []
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
            eval_set = [(X_val, y_val)]
        else:
            eval_set = None
            fit_params.pop("early_stopping_rounds", None)
            fit_params.pop("early_stopping_round", None)

        self.model = lgb.LGBMRegressor(**fit_params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
            categorical_feature=categorical_features,
        )

        if hasattr(self.model, "best_iteration_"):
            self.best_iteration = self.model.best_iteration_

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate non-negative demand predictions."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        
        preds = self.model.predict(X[self.feature_names])
        # Retail demand cannot be negative
        return np.maximum(0.0, preds)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Return dataframe of feature importances sorted descending."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        
        importances = self.model.booster_.feature_importance(importance_type=importance_type)
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp

    def save(self, filepath: str) -> None:
        """Persist model and metadata to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "best_iteration": self.best_iteration,
        }, path)
        print(f"[Model Registry] Model artifact saved to: {path}")

    @classmethod
    def load(cls, filepath: str) -> "RetailLightGBMModel":
        """Load persisted model artifact."""
        data = joblib.load(filepath)
        instance = cls(params=data.get("params"))
        instance.model = data.get("model")
        instance.feature_names = data.get("feature_names", [])
        instance.best_iteration = data.get("best_iteration")
        return instance
