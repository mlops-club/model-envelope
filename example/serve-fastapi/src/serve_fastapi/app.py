import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from serve_fastapi.models import PricePredictionRequest, PricePredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.environ.get("MODEL_PATH", "/app/model"))
    app.state.model = mlflow.pyfunc.load_model(str(model_path))
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Price Predictor API",
        description="API for predicting Runescape item prices",
        version="0.0.0",
        lifespan=lifespan,
        docs_url="/",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/predict", response_model=PricePredictionResponse)
    async def predict(request: PricePredictionRequest) -> PricePredictionResponse:
        """
        Predict the next price given a window of previous prices.
        """
        model: mlflow.pyfunc.PyFuncModel = app.state.model
        prediction = model.predict(request.to_numpy())
        return PricePredictionResponse(predicted_price=float(prediction))

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model_loaded": hasattr(app.state, "model") and app.state.model is not None,
        }

    return app
