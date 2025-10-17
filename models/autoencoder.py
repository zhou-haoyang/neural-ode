"""Autoencoder with MLP ResNet architecture for learning latent dynamics."""

import torch
import torch.nn as nn
import lightning as L
from typing import Optional, Dict, Any

from .resnet import MLPResNet


class Autoencoder(L.LightningModule):
    """Autoencoder with MLP-based ResNet encoder and decoder.
    
    The autoencoder learns a low-dimensional latent representation of
    the high-dimensional simulation state (displacement, velocity, etc.).
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ):
        """Initialize Autoencoder.
        
        Args:
            input_dim: Dimension of input state (e.g., flattened displacement field)
            latent_dim: Dimension of latent representation
            hidden_dim: Hidden dimension for ResNet blocks
            num_blocks: Number of residual blocks in encoder/decoder
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            activation: Activation function ('relu', 'gelu', 'tanh')
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Select activation function
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        act_fn = activation_map.get(activation.lower(), nn.ReLU())
        
        # Encoder: input_dim -> latent_dim
        self.encoder = MLPResNet(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            activation=act_fn,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        
        # Decoder: latent_dim -> input_dim
        self.decoder = MLPResNet(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            activation=act_fn,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Latent tensor of shape (batch_size, latent_dim)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute reconstruction loss.
        
        Args:
            batch: Dictionary containing 'state' tensor
            
        Returns:
            Dictionary with loss and metrics
        """
        x = batch["state"]
        
        # Forward pass
        z = self.encode(x)
        x_recon = self.decode(z)
        
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(x_recon, x)
        
        # Relative error
        with torch.no_grad():
            relative_error = torch.norm(x_recon - x) / torch.norm(x)
        
        return {
            "loss": recon_loss,
            "recon_loss": recon_loss,
            "relative_error": relative_error,
            "latent_norm": torch.norm(z, dim=-1).mean(),
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Dictionary containing 'state' tensor
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        metrics = self._compute_loss(batch)
        
        # Log metrics
        self.log("train/loss", metrics["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", metrics["recon_loss"], on_step=False, on_epoch=True)
        self.log("train/relative_error", metrics["relative_error"], on_step=False, on_epoch=True)
        self.log("train/latent_norm", metrics["latent_norm"], on_step=False, on_epoch=True)
        
        return metrics["loss"]
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step.
        
        Args:
            batch: Dictionary containing 'state' tensor
            batch_idx: Batch index
        """
        metrics = self._compute_loss(batch)
        
        # Log metrics
        self.log("val/loss", metrics["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon_loss", metrics["recon_loss"], on_step=False, on_epoch=True)
        self.log("val/relative_error", metrics["relative_error"], on_step=False, on_epoch=True)
        self.log("val/latent_norm", metrics["latent_norm"], on_step=False, on_epoch=True)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step.
        
        Args:
            batch: Dictionary containing 'state' tensor
            batch_idx: Batch index
        """
        metrics = self._compute_loss(batch)
        
        # Log metrics
        self.log("test/loss", metrics["loss"])
        self.log("test/recon_loss", metrics["recon_loss"])
        self.log("test/relative_error", metrics["relative_error"])
        self.log("test/latent_norm", metrics["latent_norm"])
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
