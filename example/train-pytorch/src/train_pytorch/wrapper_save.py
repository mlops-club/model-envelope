"""
Utils for saving the model wrapper instance.

It is important that these utils be in a separate file from the actual
model wrapper class, because

1. the import statements in the top of the wrapper class
   file will be executed when the model is loaded
2. and only libraries used when running model_wrapper.predict() are
   automatically included in the saved requirements.txt. So, for example,
   the `model-envelope` library will not be logged when the model wrapper is saved.
"""

import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
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
from train_pytorch.wrapper import PricePredictorWrapper


def log_price_predictor(
    model: PricePredictor,
    dataset: PriceDataset,
    model_name: str = "price_predictor",
) -> str:
    """
    Log the price predictor model using MLflow.
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
            mlflow.set_tag("git.has_uncommitted_changes", str(saved_patch).lower())

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
