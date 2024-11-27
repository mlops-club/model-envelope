import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import torch
from mlflow.pyfunc import PythonModel, PythonModelContext
from model_envelope.freeze_deps import (
    get_current_package_name,
    write_graph_to_text_file,
)
from model_envelope.git_meta import (
    save_git_patch,
    try_get_git_branch,
    try_get_git_commit,
    try_get_git_remote,
    try_get_git_user,
    try_get_git_web_url,
)

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


def get_model_config(model: torch.nn.Module) -> Dict[str, Any]:
    """Extract configuration from a trained model"""
    return {
        "window_size": model.window_size if hasattr(model, "window_size") else None,
        "hidden_size": model.embedding.out_features,
        "num_heads": model.transformer.layers[0].self_attn.num_heads,
        "num_layers": len(model.transformer.layers),
        "dropout": model.transformer.layers[0].dropout.p,
    }


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


def log_price_predictor(
    model: PricePredictor,
    dataset: PriceDataset,
    model_name: str = "price_predictor",
) -> str:
    """
    Log the price predictor model using MLflow
    """
    with mlflow.start_run() as run:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)

            # Save model config
            config_path = tmp_dir_path / "model_config.json"
            model_config = {
                "window_size": model.window_size,
            }
            config_path.write_text(json.dumps(model_config))

            # Save model state
            state_dict_path = tmp_dir_path / "model_state.pt"
            torch.save(model.state_dict(), state_dict_path)

            # Save scaler
            scaler_path = tmp_dir_path / "scaler.pt"
            torch.save(dataset.scaler, scaler_path)

            # Save dependency graphs
            full_pip_graph_fpath = tmp_dir_path / "requirements-graph-full.txt"
            write_graph_to_text_file(full_pip_graph_fpath, exclude_common_libs=False)

            model_pip_graph_fpath = tmp_dir_path / "requirements-graph-model.txt"
            write_graph_to_text_file(model_pip_graph_fpath, exclude=["model-envelope"])

            # Save git patch if there are uncommitted changes
            patch_path = tmp_dir_path / "uncommitted_changes.patch"
            artifacts = {
                "model_config": str(config_path),
                "model_state": str(state_dict_path),
                "scaler": str(scaler_path),
                "full-requirements-graph": str(full_pip_graph_fpath),
                "model-requirements-graph": str(model_pip_graph_fpath),
            }
            saved_patch = save_git_patch(patch_path)
            if saved_patch:
                artifacts["git-patch"] = str(saved_patch)

            # Set git metadata as tags
            mlflow.set_tag("current-package", get_current_package_name())
            if commit := try_get_git_commit():
                mlflow.set_tag("git.commit", commit)
            if branch := try_get_git_branch():
                mlflow.set_tag("git.branch", branch)
            if remote := try_get_git_remote():
                mlflow.set_tag("git.remote", remote)
            if user := try_get_git_user():
                mlflow.set_tag("git.user", user)
            if web_url := try_get_git_web_url():
                mlflow.set_tag("git.web_url", web_url)
            if saved_patch:
                mlflow.set_tag("git.has_uncommitted_changes", "true")

            # Create model signature
            from mlflow.models.signature import ModelSignature
            from mlflow.types.schema import Schema, TensorSpec

            input_schema = Schema(
                [TensorSpec(np.dtype(np.float64), (-1, model.window_size), "input")]
            )
            output_schema = Schema([TensorSpec(np.dtype(np.float64), (-1,), "output")])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            # Create example input that matches our expected format
            example_input = np.array(
                [[100, 101, 102, 103, 104]]
            )  # Shape: (1, window_size)

            # Validate the model works before saving
            wrapper = PricePredictorWrapper(model, dataset.scaler)
            try:
                # Test both numpy and dict inputs
                inf = wrapper.predict(None, example_input)
                inf2 = wrapper.predict(None, {"input": example_input})
            except Exception as e:
                raise RuntimeError(
                    f"Model validation failed. Model must handle both numpy arrays and "
                    f"MLflow dict inputs. Error: {str(e)}"
                ) from e

            # Log the model with MLflow
            mlflow.pyfunc.log_model(
                artifact_path=model_name,
                python_model=wrapper,  # Use the validated wrapper
                artifacts=artifacts,
                registered_model_name=model_name,
                code_paths=[str(Path(__file__).parent)],
                signature=signature,
                input_example=example_input,
            )

        return run.info.run_id
