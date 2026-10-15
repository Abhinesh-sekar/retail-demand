"""
Pydantic v2 Schemas for FastAPI Demand Forecasting & Anomaly Detection Service.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ForecastItem(BaseModel):
    date: str = Field(..., description="Forecast date (YYYY-MM-DD)")
    predicted_demand: float = Field(..., description="Forecasted sales units")


class PredictRequest(BaseModel):
    item_id: str = Field(..., json_schema_extra={"example": "FOODS_1_001"}, description="SKU identifier")
    store_id: str = Field(..., json_schema_extra={"example": "CA_1"}, description="Retail store identifier")
    horizon: int = Field(default=14, ge=1, le=28, description="Forecast horizon in days (1-28)")


class PredictResponse(BaseModel):
    item_id: str
    store_id: str
    horizon: int
    forecasts: List[ForecastItem]
    model_name: str = "LightGBM-Tweedie"


class AnomalyRecordInput(BaseModel):
    date: str
    item_id: str
    store_id: str
    actual: float
    forecast: float


class AnomalyDetectRequest(BaseModel):
    records: List[AnomalyRecordInput] = Field(
        ..., min_length=1, description="List of time series observations to evaluate"
    )
    z_threshold: Optional[float] = Field(default=3.0, ge=1.5, le=5.0)


class AnomalyItem(BaseModel):
    date: str
    item_id: str
    store_id: str
    actual: float
    forecast: float
    residual: float
    z_score: float
    anomaly_type: str
    severity: str
    recommended_action: str


class AnomalyDetectResponse(BaseModel):
    total_evaluated: int
    anomalies_flagged: int
    anomalies: List[AnomalyItem]


class MetricSummary(BaseModel):
    wape: float
    rmse: float
    mae: float
    smape: float
    bias: float


class ModelMetricsResponse(BaseModel):
    dataset: str
    backtest_folds: int
    seasonal_naive: MetricSummary
    prophet: MetricSummary
    lightgbm: MetricSummary
    relative_wape_reduction_pct: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
