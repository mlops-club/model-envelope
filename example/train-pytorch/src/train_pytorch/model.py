import numpy as np
import torch
import torch.nn as nn

from train_pytorch.constants import (
    DROPOUT,
    HIDDEN_SIZE,
    NUM_HEADS,
    NUM_LAYERS,
    POSITIONAL_ENCODING_MAX_LEN,
    WINDOW_SIZE,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = POSITIONAL_ENCODING_MAX_LEN):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[: x.size(0)]


class PricePredictor(nn.Module):
    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.window_size = window_size

        # Input embedding layer
        self.embedding = nn.Linear(1, hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, window_size, 1)

        # Embed the input
        x = self.embedding(x)  # (batch_size, window_size, hidden_size)

        # Add positional encoding
        x = x.transpose(0, 1)  # (window_size, batch_size, hidden_size)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, window_size, hidden_size)

        # Create attention mask to prevent looking at future values
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)

        # Apply transformer
        x = self.transformer(x, mask=mask)  # (batch_size, window_size, hidden_size)

        # Use only the last output
        last_output = x[:, -1, :]  # (batch_size, hidden_size)

        # Project to output
        return self.output(last_output)  # (batch_size, 1)
