"""PyTorch Lightning module for training Latent ODE models."""

from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn

from ..data.dataset import HyperelasticityDataset
from .autoencoder import Autoencoder
from .neural_ode import LatentODEModel, ODEFunc


class LatentODELightning(L.LightningModule):
    """Lightning module for training Latent ODE models.

    This model learns dynamics in latent space by:
    1. Training on sequences of observations
    2. Learning to predict future states via ODE integration
    """

    def __init__(
        self,
        autoencoder: Autoencoder,
        latent_dim: int,
        ode_hidden_dim: int = 128,
        ode_num_blocks: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        adjoint: bool = True,
        reconstruction_weight: float = 1.0,
        prediction_weight: float = 1.0,
        freeze_autoencoder: bool = False,
    ):
        """Initialize LatentODELightning.

        Args:
            autoencoder: Pretrained autoencoder model
            latent_dim: Dimension of latent space
            ode_hidden_dim: Hidden dimension for ODE function
            ode_num_blocks: Number of residual blocks in ODE function
            learning_rate: Learning rate
            weight_decay: Weight decay
            solver: ODE solver method
            rtol: Relative tolerance
            atol: Absolute tolerance
            adjoint: Use adjoint method for backprop
            reconstruction_weight: Weight for reconstruction loss
            prediction_weight: Weight for prediction loss
            freeze_autoencoder: Whether to freeze autoencoder weights
        """
        super().__init__()
        self.save_hyperparameters(ignore=["autoencoder"])

        # Extract encoder and decoder from autoencoder
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder

        # Freeze autoencoder if requested
        if freeze_autoencoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

        # Create ODE function
        self.ode_func = ODEFunc(
            latent_dim=latent_dim,
            hidden_dim=ode_hidden_dim,
            num_blocks=ode_num_blocks,
            activation=nn.ReLU(),
            time_dependent=True,
        )

        # Create Latent ODE model
        self.latent_ode = LatentODEModel(
            encoder=self.encoder,
            decoder=self.decoder,
            ode_func=self.ode_func,
            solver=solver,
            rtol=rtol,
            atol=atol,
            adjoint=adjoint,
        )

    def forward(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict trajectory from x0 over time points t.

        Args:
            x0: Initial state (batch_size, state_dim)
            t: Time points (num_times,)

        Returns:
            Predictions (num_times, batch_size, state_dim)
        """
        return self.latent_ode(x0, t)

    def _predict_sequence(
        self,
        batch: Dict[str, torch.Tensor],
        fields: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Predict state sequence from initial condition.

        Args:
            batch: Batch dictionary with field data and 'time'
            fields: List of field names to predict (default: all non-metadata fields)

        Returns:
            Tuple of (pred_dict, ref_dict) where each is a dict mapping field names to tensors:
                - field data: (batch_size, seq_len, field_dim)
        """
        times = batch["time"]  # (batch_size, seq_len)

        states = torch.cat([batch["displacement"], batch["velocity"]], dim=-1).flatten(
            start_dim=2
        )  # (batch_size, seq_len, total_dim)

        batch["states"] = states

        # Use first state as initial condition
        x0 = states[:, 0, :]  # (batch_size, total_dim)
        t = times[0, :]  # (seq_len,) - assume same for all in batch

        # Predict trajectory
        x_pred = self.latent_ode(x0, t)  # (seq_len, batch_size, total_dim)

        # Transpose to match target shape
        x_pred = x_pred.permute(1, 0, 2)  # (batch_size, seq_len, total_dim)

        num_points = batch["displacement"].shape[2]
        pred_data = x_pred.unflatten(-1, (num_points, 6))
        pred_dict = {
            "states": x_pred,
            "displacement": pred_data[..., :3],
            "velocity": pred_data[..., 3:],
        }

        return pred_dict, batch

    def _compute_loss(
        self,
        x_pred: torch.Tensor,
        x_ref: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss from predicted and reference states.

        Args:
            x_pred: Predicted states (batch_size, seq_len, state_dim)
            x_ref: Reference/ground truth states (batch_size, seq_len, state_dim)

        Returns:
            Dictionary with losses and metrics
        """
        batch_size, seq_len, state_dim = x_ref.shape

        # Reconstruction loss (MSE on all time steps)
        recon_loss = nn.functional.mse_loss(x_pred, x_ref)

        # Prediction loss (MSE on future time steps only)
        if seq_len > 1:
            pred_loss = nn.functional.mse_loss(x_pred[:, 1:, :], x_ref[:, 1:, :])
        else:
            pred_loss = recon_loss

        # Combined loss
        total_loss = (
            self.hparams.reconstruction_weight * recon_loss
            + self.hparams.prediction_weight * pred_loss
        )

        # Compute relative error
        with torch.no_grad():
            relative_error = torch.norm(x_pred - x_ref) / torch.norm(x_ref)

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "pred_loss": pred_loss,
            "relative_error": relative_error,
        }

    def _visualize_mesh(
        self,
        pred_dict: Dict[str, torch.Tensor],
        ref_dict: Dict[str, torch.Tensor],
    ) -> None:
        """Visualize predicted and reference states on mesh.

        Args:
            x_pred: Predicted states (batch_size, seq_len, state_dim)
            x_ref: Reference states (batch_size, seq_len, state_dim)
        """

        dataset: HyperelasticityDataset = self.trainer.datamodule.dataset

        u_ref = dataset.denormalize("displacement", ref_dict["displacement"].cpu())[0]
        u_pred = dataset.denormalize("displacement", pred_dict["displacement"].cpu())[0]

        self.logger.experiment.add_mesh(
            tag="predicted_mesh",
            vertices=dataset.points + u_pred,  # First sample in batch
            faces=dataset.faces.unsqueeze(0),
            global_step=self.current_epoch,
        )
        self.logger.experiment.add_mesh(
            tag="reference_mesh",
            vertices=dataset.points + u_ref,  # First sample in batch
            faces=dataset.faces.unsqueeze(0),
            global_step=self.current_epoch,
        )

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch dictionary with field data
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Predict sequence
        pred_dict, ref_dict = self._predict_sequence(batch)

        # Compute loss
        metrics = self._compute_loss(pred_dict["states"], ref_dict["states"])

        # Log metrics
        self.log(
            "train/loss", metrics["loss"], on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/recon_loss", metrics["recon_loss"], on_step=False, on_epoch=True
        )
        self.log("train/pred_loss", metrics["pred_loss"], on_step=False, on_epoch=True)
        self.log(
            "train/relative_error",
            metrics["relative_error"],
            on_step=False,
            on_epoch=True,
        )

        return metrics["loss"]

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Validation step.

        Args:
            batch: Batch dictionary with field data
            batch_idx: Batch index
        """
        # Predict sequence
        pred_dict, ref_dict = self._predict_sequence(batch)

        # Compute loss
        metrics = self._compute_loss(pred_dict["states"], ref_dict["states"])

        # Log metrics
        self.log(
            "val/loss", metrics["loss"], on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/recon_loss", metrics["recon_loss"], on_step=False, on_epoch=True)
        self.log("val/pred_loss", metrics["pred_loss"], on_step=False, on_epoch=True)
        self.log(
            "val/relative_error",
            metrics["relative_error"],
            on_step=False,
            on_epoch=True,
        )

        # Visualize mesh
        self._visualize_mesh(pred_dict, ref_dict)

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Test step.

        Args:
            batch: Batch dictionary with field data
            batch_idx: Batch index
        """
        # Predict sequence
        pred_dict, ref_dict = self._predict_sequence(batch)

        # Compute loss
        metrics = self._compute_loss(pred_dict["states"], ref_dict["states"])

        # Log metrics
        self.log("test/loss", metrics["loss"])
        self.log("test/recon_loss", metrics["recon_loss"])
        self.log("test/pred_loss", metrics["pred_loss"])
        self.log("test/relative_error", metrics["relative_error"])

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers.

        Returns:
            Dictionary with optimizer and scheduler
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

    def predict_trajectory(
        self,
        x0: torch.Tensor,
        t0: float,
        t1: float,
        num_steps: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict trajectory from t0 to t1.

        Args:
            x0: Initial state (batch_size, state_dim)
            t0: Initial time
            t1: Final time
            num_steps: Number of time steps

        Returns:
            Tuple of (time_points, predictions)
        """
        self.eval()
        with torch.no_grad():
            device = x0.device
            t = torch.linspace(t0, t1, num_steps, device=device, dtype=x0.dtype)
            x_pred = self.latent_ode(x0, t)
            # Transpose to (batch_size, num_steps, state_dim)
            x_pred = x_pred.permute(1, 0, 2)
        return t, x_pred
