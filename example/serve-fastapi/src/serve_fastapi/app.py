import os
from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from serve_fastapi.models import PricePredictionRequest, PricePredictionResponse

# Load model at startup using environment variables
model_path = Path(os.environ.get("MODEL_PATH", "/app/model"))
try:
    model = mlflow.pyfunc.load_model(str(model_path))
except Exception as e:
    raise RuntimeError(f"Failed to load model from {model_path}: {e}")

app = FastAPI(
    title="Price Predictor API",
    description="API for predicting Runescape item prices",
    version="0.1.0",
)

# Add CORS middleware
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
    try:
        prediction = model.predict({"prices": request.to_numpy()})
        return PricePredictionResponse(
            predicted_price=float(prediction),
            input_prices=request.prices,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
