"""
CLI Runner for Residual Demand Anomaly Detection.
Ingests actual sales vs model forecasts, applies Studentized Z-score thresholds,
and exports flagged supply chain anomaly reports.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.anomaly.detector import ResidualAnomalyDetector
from src.data.data_loader import load_processed_data
from src.models.lightgbm_model import RetailLightGBMModel
from src.utils.config import load_config


def main():
    print("=" * 70)
    print("  RESIDUAL-BASED DEMAND ANOMALY DETECTION ENGINE")
    print("=" * 70)
    config = load_config()

    model_path = root_dir / "models" / "lgbm_demand_model.joblib"
    encoder_path = root_dir / "models" / "categorical_encoder.joblib"
    data_path = Path(config["data"]["processed_data_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run scripts/run_train.py first.")

    print(f"[Anomaly Engine] Loading model from {model_path}...")
    model = RetailLightGBMModel.load(str(model_path))
    encoder = joblib.load(encoder_path)

    print(f"[Anomaly Engine] Loading featured sales data from {data_path}...")
    df = load_processed_data(str(data_path))

    # Evaluate the most recent 180 days
    eval_start = df["date"].max() - pd.Timedelta(days=180)
    df_eval = df[df["date"] >= eval_start].copy()
    print(f"[Anomaly Engine] Evaluating {len(df_eval):,} observations from {eval_start.date()} to {df['date'].max().date()}...")

    # Encode and predict
    df_encoded = encoder.transform(df_eval)
    feat_cols = [c for c in model.feature_names if c in df_encoded.columns]
    df_eval["forecast"] = model.predict(df_encoded[feat_cols])
    df_eval["actual"] = df_eval["sales"]

    # Run Anomaly Detection
    anomaly_cfg = config.get("anomaly", {})
    detector = ResidualAnomalyDetector(
        z_threshold=anomaly_cfg.get("z_threshold", 3.0),
        rolling_window=anomaly_cfg.get("rolling_window", 14),
        min_expected_demand=anomaly_cfg.get("min_expected_demand", 2.0),
        high_severity_threshold=anomaly_cfg.get("severity_high_threshold", 4.0),
    )

    df_annotated = detector.detect_anomalies(
        df_eval,
        actual_col="actual",
        forecast_col="forecast",
        series_cols=["item_id", "store_id"],
        date_col="date",
    )

    flagged_anomalies = detector.get_flagged_anomalies(df_annotated)

    print("\n" + "=" * 70)
    print(f"  ANOMALY DETECTION SUMMARY: {len(flagged_anomalies)} FLAGGED OF {len(df_eval)} RECORDS ({len(flagged_anomalies)/len(df_eval):.2%})")
    print("=" * 70)

    print("\nBreakdown by Anomaly Type:")
    print(flagged_anomalies["anomaly_type"].value_counts().to_string())

    print("\nBreakdown by Severity:")
    print(flagged_anomalies["severity"].value_counts().to_string())

    print("\nTop 10 High-Impact Flagged Anomalies:")
    top_cols = ["date", "item_id", "store_id", "actual", "forecast", "residual", "z_score", "anomaly_type", "severity"]
    print(flagged_anomalies[top_cols].head(10).to_string(index=False))

    # Export report
    out_csv = root_dir / "data" / "anomaly_report.csv"
    flagged_anomalies.to_csv(out_csv, index=False)
    print(f"\n[Anomaly Engine] Full anomaly report written to: {out_csv}")


if __name__ == "__main__":
    main()
