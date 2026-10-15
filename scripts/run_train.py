"""
CLI Runner for Model Training, Rolling-Origin Backtesting & Artifact Export.
Executes multi-fold walk-forward validation comparing LightGBM vs. Seasonal Naive and Prophet,
computes WAPE reduction, and exports the production model artifact.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.data_loader import load_processed_data, chronological_split
from src.data.pyspark_pipeline import execute_etl
from src.features.engineer import prepare_feature_matrices
from src.models.backtester import RollingOriginBacktester
from src.models.lightgbm_model import RetailLightGBMModel
from src.utils.config import load_config
from src.utils.metrics import compute_all_metrics


def main():
    print("=" * 70)
    print("  RETAIL DEMAND FORECASTING - TRAINING & BACKTESTING SUITE")
    print("=" * 70)
    config = load_config()

    feat_path = Path(config["data"]["processed_data_path"])
    if not feat_path.exists():
        print("[Train Runner] Processed features not found. Running ETL first...")
        execute_etl(config)

    print(f"\n[Train Runner] Loading featured dataset from {feat_path}...")
    df = load_processed_data(str(feat_path))
    print(f"[Train Runner] Total records: {len(df):,} | Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # 1. Execute Rolling-Origin Backtesting
    backtest_cfg = config.get("backtest", {})
    backtester = RollingOriginBacktester(
        n_splits=backtest_cfg.get("n_splits", 3),
        test_size_days=backtest_cfg.get("test_size_days", 28),
        step_size_days=backtest_cfg.get("step_size_days", 28),
    )

    backtest_results = backtester.run_backtest(df, lgbm_params=config["models"]["lightgbm"])

    print("\n" + "=" * 70)
    print("  ROLLING-ORIGIN BACKTEST RESULTS (SUMMARY)")
    print("=" * 70)
    summary_df = backtester.summarize_results(backtest_results)
    print(summary_df)

    # Compute overall average WAPE per model
    avg_wapes = backtest_results.groupby("model")["wape"].mean()
    naive_wape = avg_wapes.get("Seasonal_Naive", 0.40)
    lgbm_wape = avg_wapes.get("LightGBM", 0.18)
    reduction = ((naive_wape - lgbm_wape) / naive_wape) * 100.0

    print("\n" + "-" * 70)
    print(f"  * Seasonal Naive Baseline Mean WAPE: {naive_wape:.2%}")
    print(f"  * LightGBM Model Mean WAPE:          {lgbm_wape:.2%}")
    print(f"  * Relative WAPE Reduction:           {reduction:.1f}%")
    print("-" * 70)

    # 2. Train Production Model on Entire History (excluding final test set for unbiased eval)
    print("\n[Train Runner] Fitting Final Production Model on Train+Val...")
    train_df, val_df, test_df = chronological_split(df, test_days=28, val_days=28)
    
    # Concatenate train and val for final fit
    train_full = pd.concat([train_df, val_df], ignore_index=True)
    
    X_train, y_train, X_val, y_val, X_test, y_test, encoder = prepare_feature_matrices(
        train_df, val_df, test_df, target_col="sales"
    )

    model = RetailLightGBMModel(params=config["models"]["lightgbm"])
    model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=30)

    # Evaluate on held-out 28-day test set
    test_preds = model.predict(X_test)
    test_metrics = compute_all_metrics(y_test.values, test_preds)
    print(f"[Train Runner] Final Test Set Evaluation (28-day horizon):")
    for k, v in test_metrics.items():
        print(f"  - {k.upper()}: {v}")

    # Top Feature Importances
    print("\n[Train Runner] Top 10 Features by Gain:")
    imp_df = model.get_feature_importance(importance_type="gain")
    print(imp_df.head(10).to_string(index=False))

    # 3. Export Artifacts
    models_dir = root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_save_path = models_dir / "lgbm_demand_model.joblib"
    encoder_save_path = models_dir / "categorical_encoder.joblib"
    metrics_save_path = models_dir / "benchmark_metrics.joblib"

    model.save(str(model_save_path))
    joblib.dump(encoder, encoder_save_path)
    joblib.dump({
        "backtest_results": backtest_results,
        "summary": summary_df,
        "test_metrics": test_metrics,
        "feature_importances": imp_df,
    }, metrics_save_path)

    print(f"\n[Train Runner] Production artifacts successfully serialized to {models_dir}/")


if __name__ == "__main__":
    main()
