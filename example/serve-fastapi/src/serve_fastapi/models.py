import numpy as np
from pydantic import BaseModel


class PricePredictionRequest(BaseModel):
    """Request model for price prediction endpoint"""

    # Last N prices, where N is the model's window size
    prices: list[float] | list[list[float]]

    def to_numpy(self) -> np.ndarray:
        # return np.array(self.prices).reshape(-1, 1)

        # multiple inputs
        if isinstance(self.prices[0], list):
            return np.array(self.prices)

        # single input
        return np.array([self.prices])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prices": [
                        100.0,
                        102.5,
                        101.8,
                        103.2,
                        102.9,
                    ]
                }
            ]
        }
    }


class PricePredictionResponse(BaseModel):
    """Response model for price prediction endpoint"""

    predicted_price: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_price": 103.5,
                }
            ]
        }
    }
