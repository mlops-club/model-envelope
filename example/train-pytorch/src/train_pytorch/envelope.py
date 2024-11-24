import json
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
        config = torch.load(context.artifacts["model_config"])
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
        self, context: PythonModelContext, model_input: np.ndarray
    ) -> np.ndarray:
        """
        Predict using the wrapped model.

        Args:
            context: MLflow model context
            model_input: Array of shape (batch_size, window_size) containing price sequences

        Returns:
            Array of shape (batch_size,) containing predicted prices
        """
        self.model.eval()
        with torch.no_grad():
            # Scale the input
            scaled_input = self.scaler.transform(model_input.reshape(-1, 1)).reshape(
                model_input.shape[0], -1
            )
            X = torch.FloatTensor(scaled_input).to(self.device)

            # Get prediction
            scaled_pred = self.model(X)

            # Convert back to original scale
            predictions = self.scaler.inverse_transform(scaled_pred.cpu())

            return predictions.reshape(-1)


def get_model_config(model: torch.nn.Module) -> Dict[str, Any]:
    return predict_next_day(
        model=self.model,
        last_window=model_input,
        scaler=self.scaler,
        device=self.device,
    )


def predict_next_day(
    model: PricePredictor,
    last_window: np.ndarray,
    scaler: Any,
    device: Optional[str] = None,
) -> float:
    """
    Predict the next day's price
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    with torch.no_grad():
        # Scale the input
        scaled_window = scaler.transform(last_window.reshape(-1, 1))
        X = (
            torch.FloatTensor(scaled_window).unsqueeze(0).to(device)
        )  # Add batch dimension

        # Get prediction
        scaled_pred = model(X)

        # Convert back to original scale
        prediction = scaler.inverse_transform(scaled_pred.cpu())[0][0]

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
            if saved_patch := save_git_patch(patch_path):
                artifacts = {
                    "model_config": str(config_path),
                    "model_state": str(state_dict_path),
                    "scaler": str(scaler_path),
                    "full-requirements-graph": str(full_pip_graph_fpath),
                    "model-requirements-graph": str(model_pip_graph_fpath),
                    "git-patch": str(saved_patch),
                }
            else:
                artifacts = {
                    "model_config": str(config_path),
                    "model_state": str(state_dict_path),
                    "scaler": str(scaler_path),
                    "full-requirements-graph": str(full_pip_graph_fpath),
                    "model-requirements-graph": str(model_pip_graph_fpath),
                }

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

            # Create an example input that matches our expected format
            example_input = {"data": np.array([314, 315, 316, 317, 318])}

            # Log the model with MLflow
            mlflow.pyfunc.log_model(
                artifact_path=model_name,
                python_model=PricePredictorWrapper(model, dataset.scaler),
                artifacts=artifacts,
                registered_model_name=model_name,
                code_paths=[str(Path(__file__).parent)],
                input_example=example_input,
            )

        return run.info.run_id

        return run.info.run_id
