from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from train_pytorch.constants import (
    RUNESCAPE_ITEM_NAMES_DATASET_FPATH,
    RUNESCAPE_ITEM_PRICES_DATASET_FPATH,
    WINDOW_SIZE,
)


class PriceDataset(Dataset):
    def __init__(self, prices: pd.Series, window_size: int = WINDOW_SIZE):
        """
        Args:
            prices: Time series of prices
            window_size: Number of days to use for prediction
        """
        self.window_size = window_size

        # Scale the data
        scaled_data, self.scaler = scale_data(prices.values)

        # Create sequences
        self.X, self.y = create_sequences(scaled_data, window_size)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def inverse_transform(self, scaled_value: torch.Tensor) -> np.ndarray:
        """Convert scaled values back to original scale"""
        return self.scaler.inverse_transform(scaled_value.reshape(-1, 1))

    @classmethod
    def from_dataset_fpath(
        cls,
        item_id: int,
        prices_fpath: Path = RUNESCAPE_ITEM_PRICES_DATASET_FPATH,
        names_fpath: Path = RUNESCAPE_ITEM_NAMES_DATASET_FPATH,
        window_size: int = WINDOW_SIZE,
    ) -> "PriceDataset":
        """Create dataset from file paths"""
        # Read and clean dataframes
        prices_df = clean_prices_df(pl.read_csv(prices_fpath))
        names_df = clean_names_df(pl.read_csv(names_fpath))

        # Merge and prepare time series
        merged_df = merge_price_names_dataframes(prices_df, names_df)
        prices = prepare_time_series(merged_df, item_id)

        return cls(prices, window_size=window_size)

    def get_scaler_params(self) -> Dict[str, Any]:
        """Get the parameters of the scaler for saving"""
        if hasattr(self.scaler, "get_params"):
            return self.scaler.get_params()
        elif hasattr(self.scaler, "__dict__"):
            return self.scaler.__dict__
        else:
            raise ValueError("Scaler does not have parameters that can be extracted")


def clean_names_df(names_df: pl.DataFrame) -> pl.DataFrame:
    """Clean the names dataframe by converting Name_ID to proper format"""
    return names_df.with_columns(
        names_df["Name_ID"]
        .cast(pl.Utf8)  # Ensure it's a string
        .str.strip_chars()  # Remove leading/trailing spaces
        .cast(pl.Int64)  # Convert to integer
    )


def clean_prices_df(prices_df: pl.DataFrame) -> pl.DataFrame:
    """Clean the prices dataframe by converting date to proper format"""
    return prices_df.with_columns(
        pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
    )


def merge_price_names_dataframes(
    prices_df: pl.DataFrame, names_df: pl.DataFrame
) -> pl.DataFrame:
    """Merge prices and names dataframes"""
    return prices_df.join(names_df, left_on="id", right_on="Name_ID", how="inner")


def prepare_time_series(df: pl.DataFrame, item_id: int) -> pd.Series:
    """
    Prepare time series for a specific item

    Args:
        df: Merged dataframe with prices and names
        item_id: ID of the item to get prices for

    Returns:
        Pandas Series with datetime index and prices
    """
    # Filter for specific item and sort by date
    item_df = df.filter(pl.col("id") == item_id).sort("date").select(["date", "price"])

    # Convert to pandas series
    return pd.Series(
        item_df.get_column("price").to_numpy(),
        index=pd.DatetimeIndex(item_df.get_column("date").to_numpy()),
    )


def scale_data(data: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """Scale data to [0,1] range and return scaler"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler


def create_sequences(
    data: np.ndarray, window_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create sequences for training from scaled data"""
    X = []
    y = []

    for i in range(len(data) - window_size):
        X.append(data[i : (i + window_size)])
        y.append(data[i + window_size])

    return (torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)))
