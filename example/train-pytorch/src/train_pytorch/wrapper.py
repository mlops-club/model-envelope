import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from mlflow.pyfunc import PythonModel, PythonModelContext

from train_pytorch.dataset import PriceDataset
from train_pytorch.model import PricePredictor

# This line is needed to see full tracebacks in MLflow when saving a model silently fails.
# --specifically when it fails to detect the minimal dependencies needed for inference
# which are derived by running an example_input through the predict method.
logging.getLogger("mlflow").setLevel(logging.DEBUG)


class PricePredictorWrapper(PythonModel):
    def __init__(
        self,
        model: Optional[PricePredictor] = None,
        scaler: Optional[Any] = None,
    ):
        self.model = model
        self.scaler = scaler
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_context(self, context: PythonModelContext) -> None:
        """Load the model and scaler from the saved artifacts"""
        # Load config and state dict
        with open(context.artifacts["model_config"]) as f:
            config = json.load(f)
        state_dict = torch.load(
            context.artifacts["model_state"], map_location=self.device
        )
        self.scaler = torch.load(context.artifacts["scaler"])

        # Initialize and load model
        self.model = PricePredictor(**config)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self, context: PythonModelContext, model_input: Dict | np.ndarray
    ) -> np.ndarray:
        """
        Predict using the wrapped model.

        Args:
            context: MLflow model context
            model_input: Either:
                - numpy array of shape (batch_size, window_size)
                - dict with 'inputs' key containing array of shape (batch_size, window_size)

        Returns:
            Array of shape (batch_size,) containing predicted prices
        """
        from rich import print

        print(model_input)

        self.model.eval()
        with torch.no_grad():
            # Convert dict input to numpy array if needed
            if isinstance(model_input, dict):
                model_input = np.array(model_input["input"])

            # Scale the input
            scaled_input = self.scaler.transform(model_input.reshape(-1, 1))
            # Reshape to (batch_size, window_size, 1)
            scaled_input = scaled_input.reshape(model_input.shape[0], -1, 1)
            X = torch.FloatTensor(scaled_input).to(self.device)

            # Get prediction
            scaled_pred = self.model(X)

            # Convert back to original scale
            predictions = self.scaler.inverse_transform(scaled_pred.cpu())

            return predictions.reshape(-1)


def predict_next_day(
    model: PricePredictor,
    dataset: PriceDataset,
    last_window: np.ndarray,
    device: Optional[str] = None,
) -> float:
    """
    Predict the next day's price
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    with torch.no_grad():
        # Scale the input using dataset's scaler
        scaled_window = dataset.scaler.transform(last_window.reshape(-1, 1))
        X = torch.FloatTensor(scaled_window).unsqueeze(0).to(device)

        # Get prediction
        scaled_pred = model(X)

        # Convert back to original scale
        prediction = dataset.scaler.inverse_transform(scaled_pred.cpu())[0][0]

    return prediction


def get_model_config(model: torch.nn.Module) -> Dict[str, Any]:
    """Extract configuration from a trained model"""
    return {
        "window_size": model.window_size if hasattr(model, "window_size") else None,
        "hidden_size": model.embedding.out_features,
        "num_heads": model.transformer.layers[0].self_attn.num_heads,
        "num_layers": len(model.transformer.layers),
        "dropout": model.transformer.layers[0].dropout.p,
    }
