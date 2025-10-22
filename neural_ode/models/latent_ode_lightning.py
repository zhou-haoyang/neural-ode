"""PyTorch Lightning module for training Latent ODE models."""

from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pyvista as pv

from ..data.dataset import HyperelasticityDataset
from .autoencoder import Autoencoder
from .neural_ode import LatentODEModel, ODEFunc


def create_animation_frames(
    results: dict,
    show_error: bool = True,
    show_reference: bool = True,
    resolution: tuple = (800, 600),
) -> np.ndarray:
    """Create animation frames for TensorBoard video logging.

    Args:
        results: Dictionary from integrate_full_trajectory
        show_error: Whether to color mesh by error magnitude
        show_reference: Whether to show reference mesh alongside prediction
        resolution: (width, height) of output frames

    Returns:
        Video frames as numpy array of shape (T, H, W, C) where T is number of timesteps
    """
    points = results["points"]
    faces = results["faces"]
    pred_disp = results["pred_displacement"]
    ref_disp = results["ref_displacement"]
    times = results["times"]

    n_timesteps = len(times)

    # Convert faces to PyVista format (prepend count)
    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).flatten()

    # Create off-screen plotter
    if show_reference:
        plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=resolution)
    else:
        plotter = pv.Plotter(off_screen=True, window_size=resolution)

    # Set up initial camera position (will be reused for all frames)
    camera_position = None
    camera_position_ref = None

    frames = []

    # Function to render frame
    def render_frame(frame_idx: int):
        nonlocal camera_position, camera_position_ref

        plotter.clear()

        # Deformed points
        pred_points_deformed = points + pred_disp[frame_idx]
        ref_points_deformed = points + ref_disp[frame_idx]

        # Create meshes
        pred_mesh = pv.PolyData(pred_points_deformed, pv_faces)
        ref_mesh = pv.PolyData(ref_points_deformed, pv_faces)

        if show_error:
            # Compute per-point error
            error = np.linalg.norm(pred_disp[frame_idx] - ref_disp[frame_idx], axis=1)
            pred_mesh["error"] = error

        # Plot prediction
        if show_reference:
            plotter.subplot(0, 0)

        if show_error:
            plotter.add_mesh(
                pred_mesh,
                scalars="error",
                cmap="hot",
                show_edges=True,
                edge_color="black",
                line_width=0.5,
                scalar_bar_args={"title": "Error"},
            )
        else:
            plotter.add_mesh(
                pred_mesh,
                color="lightblue",
                show_edges=True,
                edge_color="black",
                line_width=0.5,
            )

        plotter.add_text(
            f"Prediction (t={times[frame_idx]:.3f}s)",
            position="upper_edge",
            font_size=12,
        )

        # Set camera only on first frame, then reuse
        if camera_position is None:
            plotter.view_isometric()
            camera_position = plotter.camera_position
        else:
            plotter.camera_position = camera_position

        # Plot reference if requested
        if show_reference:
            plotter.subplot(0, 1)
            plotter.add_mesh(
                ref_mesh,
                color="lightgreen",
                show_edges=True,
                edge_color="black",
                line_width=0.5,
            )
            plotter.add_text(
                f"Reference (t={times[frame_idx]:.3f}s)",
                position="upper_edge",
                font_size=12,
            )

            # Set camera only on first frame, then reuse
            if camera_position_ref is None:
                plotter.view_isometric()
                camera_position_ref = plotter.camera_position
            else:
                plotter.camera_position = camera_position_ref

        # Capture frame as image
        plotter.render()
        img = plotter.screenshot(return_img=True)
        return img

    # Generate all frames
    for frame_idx in tqdm(range(n_timesteps), desc="Rendering frames"):
        frame = render_frame(frame_idx)
        frames.append(frame)

    plotter.close()

    # Stack frames into video array (T, H, W, C)
    video = np.stack(frames, axis=0)

    return video


def create_animation(
    results: dict,
    output_path: Optional[str] = None,
    framerate: int = 10,
    show_error: bool = True,
    show_reference: bool = True,
):
    """Create PyVista animation of predicted vs reference trajectories.

    Args:
        results: Dictionary from integrate_full_trajectory
        output_path: Path to save animation (e.g., 'animation.gif' or 'animation.mp4')
        framerate: Frames per second for animation
        show_error: Whether to color mesh by error magnitude
        show_reference: Whether to show reference mesh alongside prediction
    """
    points = results["points"]
    faces = results["faces"]
    pred_disp = results["pred_displacement"]
    ref_disp = results["ref_displacement"]
    times = results["times"]

    n_timesteps = len(times)

    # Convert faces to PyVista format (prepend count)
    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).flatten()

    # Create plotter
    if show_reference:
        plotter = pv.Plotter(shape=(1, 2), off_screen=output_path is not None)
    else:
        plotter = pv.Plotter(off_screen=output_path is not None)

    # Set up initial camera position (will be reused for all frames)
    camera_position = None
    camera_position_ref = None

    # Function to update frame
    def update_frame(frame_idx: int):
        nonlocal camera_position, camera_position_ref

        plotter.clear()

        # Deformed points
        pred_points_deformed = points + pred_disp[frame_idx]
        ref_points_deformed = points + ref_disp[frame_idx]

        # Create meshes
        pred_mesh = pv.PolyData(pred_points_deformed, pv_faces)
        ref_mesh = pv.PolyData(ref_points_deformed, pv_faces)

        if show_error:
            # Compute per-point error
            error = np.linalg.norm(pred_disp[frame_idx] - ref_disp[frame_idx], axis=1)
            pred_mesh["error"] = error

        # Plot prediction
        if show_reference:
            plotter.subplot(0, 0)

        if show_error:
            plotter.add_mesh(
                pred_mesh,
                scalars="error",
                cmap="hot",
                show_edges=True,
                edge_color="black",
                line_width=0.5,
                scalar_bar_args={"title": "Error"},
            )
        else:
            plotter.add_mesh(
                pred_mesh,
                color="lightblue",
                show_edges=True,
                edge_color="black",
                line_width=0.5,
            )

        plotter.add_text(
            f"Prediction (t={times[frame_idx]:.3f}s)",
            position="upper_edge",
            font_size=12,
        )

        # Set camera only on first frame, then reuse
        if camera_position is None:
            plotter.view_isometric()
            camera_position = plotter.camera_position
        else:
            plotter.camera_position = camera_position

        # Plot reference if requested
        if show_reference:
            plotter.subplot(0, 1)
            plotter.add_mesh(
                ref_mesh,
                color="lightgreen",
                show_edges=True,
                edge_color="black",
                line_width=0.5,
            )
            plotter.add_text(
                f"Reference (t={times[frame_idx]:.3f}s)",
                position="upper_edge",
                font_size=12,
            )

            # Set camera only on first frame, then reuse
            if camera_position_ref is None:
                plotter.view_isometric()
                camera_position_ref = plotter.camera_position
            else:
                plotter.camera_position = camera_position_ref

    # Create animation
    if output_path:
        print(f"\nGenerating animation: {output_path}")
        plotter.open_gif(output_path, fps=framerate)

        for frame_idx in tqdm(range(n_timesteps), desc="Rendering frames"):
            update_frame(frame_idx)
            plotter.write_frame()

        plotter.close()
        print(f"Animation saved to: {output_path}")
    else:
        # Interactive animation
        print("\nStarting interactive animation (close window to exit)")
        plotter.show(auto_close=False)

        for frame_idx in range(n_timesteps):
            update_frame(frame_idx)
            plotter.render()
            plotter.update(force_redraw=True)

        plotter.close()


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

    def _get_input_states(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Get input states from batch by concatenating displacement and velocity.

        Args:
            batch: Batch dictionary with field data

        Returns:
            States tensor of shape (batch_size, seq_len, state_dim)
        """
        states = torch.cat([batch["displacement"], batch["velocity"]], dim=-1).flatten(
            start_dim=-2
        )  # (batch_size, seq_len, total_dim)
        return states

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

        states = self._get_input_states(batch)  # (batch_size, seq_len, total_dim)

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

    def _visualize_trajectory(self) -> None:
        """Visualize predicted trajectory as TensorBoard video."""
        dataset: HyperelasticityDataset = self.trainer.datamodule.dataset

        # Integrate over full trajectory
        first_sample = dataset[0]
        states = self._get_input_states(first_sample)  # (seq_len, state_dim)

        x0 = states[[0], :].to(self.device)  # (1, state_dim)
        t = dataset.time_steps.to(self.device)  # (n_timesteps,)
        x_pred = self.latent_ode(x0, t)  # (n_timesteps, 1, state_dim)
        x_pred = x_pred.squeeze(1).unflatten(
            -1, (dataset.points.shape[0], 6)
        )  # (n_timesteps, n_points, 6)
        pred_displacement = dataset.denormalize("displacement", x_pred[..., :3].cpu())
        pred_velocity = dataset.denormalize(
            "velocity", x_pred[..., 3:].cpu()
        )  # (n_timesteps, n_points, 3)

        # Create results dict
        results = {
            "pred_displacement": pred_displacement.numpy(),
            "pred_velocity": pred_velocity.numpy(),
            "ref_displacement": dataset.field_data[
                "displacement"
            ].numpy(),  # (n_timesteps, n_points, 3)
            "ref_velocity": dataset.field_data[
                "velocity"
            ].numpy(),  # (n_timesteps, n_points, 3)
            "times": dataset.time_steps.numpy(),
            "points": dataset.points.numpy(),
            "faces": dataset.faces.numpy(),
        }

        # Generate video frames
        video_frames = create_animation_frames(
            results,
            show_error=True,
            show_reference=True,
            resolution=(800, 600),  # Higher resolution for better quality
        )

        # Convert to TensorBoard format: (N, T, C, H, W)
        # video_frames is (T, H, W, C), we need to add batch dimension and reorder
        video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).unsqueeze(0)
        # Shape: (1, T, C, H, W)

        # Log to TensorBoard
        self.logger.experiment.add_video(
            tag="trajectory/prediction_vs_reference",
            vid_tensor=video_tensor,
            global_step=self.current_epoch,
            fps=10,
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

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch."""
        # Visualize trajectory every 50 epochs
        if self.current_epoch > 0 and self.current_epoch % 50 == 0:
            # if self.current_epoch % 50 == 0:
            self._visualize_trajectory()

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
