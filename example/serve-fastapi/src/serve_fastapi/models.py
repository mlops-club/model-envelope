from typing import List

import numpy as np
from pydantic import BaseModel


class PricePredictionRequest(BaseModel):
    """Request model for price prediction endpoint"""

    prices: List[float]  # Last N prices, where N is the model's window size

    def to_numpy(self) -> np.ndarray:
        return np.array(self.prices).reshape(-1, 1)

    model_config = {
        "json_schema_extra": {
            "examples": [{"prices": [100.0, 102.5, 101.8, 103.2, 102.9]}]
        }
    }


class PricePredictionResponse(BaseModel):
    """Response model for price prediction endpoint"""

    predicted_price: float
    input_prices: List[float]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_price": 103.5,
                    "input_prices": [100.0, 102.5, 101.8, 103.2, 102.9],
                }
            ]
        }
    }
