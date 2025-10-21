"""MLP-based ResNet block implementation."""

import torch
import torch.nn as nn
from typing import Optional


class ResidualBlock(nn.Module):
    """Residual block with MLP architecture.

    Implements: output = activation(F(x) + x) where F is a 2-layer MLP.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ):
        """Initialize ResidualBlock.

        Args:
            dim: Input and output dimension
            hidden_dim: Hidden layer dimension (defaults to dim)
            activation: Activation function
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim

        layers = []

        # First linear layer
        layers.append(nn.Linear(dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Second linear layer
        layers.append(nn.Linear(hidden_dim, dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))

        self.block = nn.Sequential(*layers)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch_size, dim)

        Returns:
            Output tensor of shape (batch_size, dim)
        """
        return self.activation(self.block(x) + x)


class MLPResNet(nn.Module):
    """MLP with residual connections."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ):
        """Initialize MLPResNet.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Hidden dimension for residual blocks
            num_blocks: Number of residual blocks
            activation: Activation function
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Project to hidden dimension
        x = self.activation(self.input_proj(x))

        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Project to output dimension
        x = self.output_proj(x)

        return x
