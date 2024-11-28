import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
from fastapi import FastAPI

from serve_fastapi.models import PricePredictionRequest, PricePredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Read the model from disk on startup (this assumes it is baked into the Docker image)."""
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

    @app.post("/predict", response_model=PricePredictionResponse)
    async def predict(request: PricePredictionRequest) -> PricePredictionResponse:
        """Predict the next price given a window of previous prices."""
        model: mlflow.pyfunc.PyFuncModel = app.state.model
        prediction = model.predict(request.to_numpy())
        return PricePredictionResponse(predicted_price=float(prediction))

    return app
