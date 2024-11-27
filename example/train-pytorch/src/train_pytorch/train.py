from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_pytorch.constants import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WINDOW_SIZE,
)
from train_pytorch.dataset import PriceDataset
from train_pytorch.model import PricePredictor
from train_pytorch.wrapper import predict_next_day
from train_pytorch.wrapper_save import log_price_predictor


def train_model(
    dataset: PriceDataset,
    window_size: int = WINDOW_SIZE,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    device: Optional[str] = None,
) -> PricePredictor:
    """
    Train a price prediction model

    Args:
        dataset: Dataset to train on
        window_size: Number of days to use for prediction
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on ('cuda' or 'cpu')
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = PricePredictor(window_size=window_size)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

    return model


def train_and_save_model(
    item_id: int,
    model_name: Optional[str] = None,
    epochs: int = EPOCHS,
    device: Optional[str] = None,
) -> Tuple[PricePredictor, PriceDataset, str]:
    """
    Train and save a model for a specific item

    Args:
        item_id: ID of the item to train model for
        model_name: Name for saving the model (defaults to f"item_{item_id}")
        epochs: Number of training epochs
        device: Device to train on

    Returns:
        Tuple of (trained model, dataset, run_id)
    """
    # Create dataset
    dataset = PriceDataset.from_dataset_fpath(item_id=item_id)

    # Train model
    model = train_model(dataset, epochs=epochs, device=device)

    # Log model with MLflow
    if model_name is None:
        model_name = f"item_{item_id}"

    run_id = log_price_predictor(model, dataset, model_name)

    return model, dataset, run_id


if __name__ == "__main__":
    # Example: Train model for Dragon dagger (ID: 1215)
    ITEM_ID = 1215
    model, dataset, run_id = train_and_save_model(ITEM_ID, epochs=5)
    print(f"Model saved with run ID: {run_id}")

    # Get last window of prices for prediction
    last_window = dataset.inverse_transform(dataset.X[-1])
    next_price = predict_next_day(model, dataset, last_window)

    print(f"\nLast known prices: {last_window.flatten()}")
    print(f"Predicted next price: {next_price:.2f}")
