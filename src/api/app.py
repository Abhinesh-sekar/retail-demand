"""
FastAPI Application Entry Point.
Initializes application state, loads models and feature registries on startup,
and configures OpenAPI / Swagger documentation.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.features.encoders import CategoricalLabelEncoder
from src.models.lightgbm_model import RetailLightGBMModel
from src.utils.config import get_project_root, load_config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: preload models and artifacts on startup."""
    root = get_project_root()
    model_path = root / "models" / "lgbm_demand_model.joblib"
    encoder_path = root / "models" / "categorical_encoder.joblib"
    data_path = root / "data" / "processed" / "featured_sales.parquet"

    print("[FastAPI Lifespan] Initializing application state...")

    # Load Model
    if model_path.exists():
        try:
            app.state.model = RetailLightGBMModel.load(str(model_path))
            print(f"[FastAPI Lifespan] Successfully loaded LightGBM model from {model_path}")
        except Exception as e:
            print(f"[FastAPI Lifespan] Error loading model: {e}")
            app.state.model = None
    else:
        print("[FastAPI Lifespan] Model file not found; initializing empty model state.")
        app.state.model = None

    # Load Encoder
    if encoder_path.exists():
        try:
            import joblib
            app.state.encoder = joblib.load(encoder_path)
            print("[FastAPI Lifespan] Categorical encoder loaded.")
        except Exception:
            app.state.encoder = None
    else:
        app.state.encoder = None

    # Load sample features for inference
    if data_path.exists():
        try:
            df = pd.read_parquet(data_path)
            app.state.latest_features = df.tail(5000).copy()
            print(f"[FastAPI Lifespan] Preloaded {len(app.state.latest_features)} latest feature rows.")
        except Exception:
            app.state.latest_features = None
    else:
        app.state.latest_features = None

    yield
    print("[FastAPI Lifespan] Shutting down application...")


def create_app() -> FastAPI:
    """Create and configure FastAPI instance."""
    app = FastAPI(
        title="Retail Demand Forecasting & Anomaly Detection API (2026)",
        description=(
            "Production-grade REST API serving LightGBM SKU-store daily demand forecasts "
            "and residual-based supply chain anomaly detection (demand spikes & stockout plunges)."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
