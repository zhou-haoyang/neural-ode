"""Neural ODE implementation for learning latent dynamics."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint, odeint_adjoint

    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. Install with: pip install torchdiffeq")

from .resnet import MLPResNet


class ODEFunc(nn.Module):
    """ODE function f(t, z) that defines dz/dt = f(t, z).

    This is the dynamics model that learns how the latent state evolves over time.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        use_batch_norm: bool = False,  # Often disabled in ODE functions
        time_dependent: bool = True,
    ):
        """Initialize ODE function.

        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for MLP
            num_blocks: Number of residual blocks
            activation: Activation function
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            time_dependent: Whether to include time as input
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.time_dependent = time_dependent

        # Input dimension: latent_dim + 1 if time-dependent, else latent_dim
        input_dim = latent_dim + 1 if time_dependent else latent_dim

        # MLP to compute dz/dt
        self.mlp = MLPResNet(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt = f(t, z).

        Args:
            t: Time tensor of shape () or (1,)
            z: Latent state tensor of shape (batch_size, latent_dim)

        Returns:
            Time derivative dz/dt of shape (batch_size, latent_dim)
        """
        if self.time_dependent:
            # Expand time to match batch size
            batch_size = z.shape[0]
            t_expanded = t.expand(batch_size, 1)

            # Concatenate time with latent state
            tz = torch.cat([t_expanded, z], dim=-1)
            dzdt = self.mlp(tz)
        else:
            dzdt = self.mlp(z)

        return dzdt


class NeuralODE(nn.Module):
    """Neural ODE module for solving ODEs with neural network dynamics.

    Given an initial state z0 at time t0, solves the ODE:
        dz/dt = f(t, z)
    to obtain z(t) for any t > t0.
    """

    def __init__(
        self,
        ode_func: ODEFunc,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        adjoint: bool = True,
    ):
        """Initialize NeuralODE.

        Args:
            ode_func: ODE function defining dynamics
            solver: ODE solver method ('dopri5', 'rk4', 'euler', 'adaptive_heun')
            rtol: Relative tolerance for adaptive solvers
            atol: Absolute tolerance for adaptive solvers
            adjoint: Whether to use adjoint method for backpropagation
        """
        super().__init__()

        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError(
                "torchdiffeq is required for Neural ODE. "
                "Install with: pip install torchdiffeq"
            )

        self.ode_func = ode_func
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint

        # Choose integration method
        self.odeint = odeint_adjoint if adjoint else odeint

    def forward(self, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Solve ODE from initial state z0 over time points t.

        Args:
            z0: Initial state of shape (batch_size, latent_dim)
            t: Time points of shape (num_times,), must include t[0] = t0

        Returns:
            Solutions of shape (num_times, batch_size, latent_dim)
        """
        # Solve ODE
        zt = self.odeint(
            self.ode_func,
            z0,
            t,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver,
        )

        return zt

    def integrate(
        self,
        z0: torch.Tensor,
        t0: float,
        t1: float,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate from t0 to t1.

        Args:
            z0: Initial state of shape (batch_size, latent_dim)
            t0: Initial time
            t1: Final time
            num_steps: Number of time steps (if None, solver adaptive)

        Returns:
            Tuple of (time_points, solutions)
                - time_points: shape (num_times,)
                - solutions: shape (num_times, batch_size, latent_dim)
        """
        device = z0.device

        if num_steps is None:
            # Just evaluate at endpoints for adaptive solvers
            t = torch.tensor([t0, t1], device=device, dtype=z0.dtype)
        else:
            # Create uniform time grid
            t = torch.linspace(t0, t1, num_steps, device=device, dtype=z0.dtype)

        zt = self.forward(z0, t)

        return t, zt


class LatentODEModel(nn.Module):
    """Complete latent ODE model combining encoder, ODE, and decoder.

    This model:
    1. Encodes observations to latent space
    2. Evolves latent state using Neural ODE
    3. Decodes latent state back to observation space
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        ode_func: ODEFunc,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        adjoint: bool = True,
    ):
        """Initialize LatentODEModel.

        Args:
            encoder: Encoder module (e.g., from Autoencoder)
            decoder: Decoder module (e.g., from Autoencoder)
            ode_func: ODE function defining latent dynamics
            solver: ODE solver method
            rtol: Relative tolerance
            atol: Absolute tolerance
            adjoint: Whether to use adjoint method
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.neural_ode = NeuralODE(
            ode_func=ode_func,
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
        """Predict trajectory starting from x0 over time points t.

        Args:
            x0: Initial observation of shape (batch_size, state_dim)
            t: Time points of shape (num_times,)

        Returns:
            Predicted observations of shape (num_times, batch_size, state_dim)
        """
        # Encode initial state
        z0 = self.encoder(x0)  # (batch_size, latent_dim)

        # Evolve in latent space
        zt = self.neural_ode(z0, t)  # (num_times, batch_size, latent_dim)

        # Decode each time point
        num_times, batch_size, latent_dim = zt.shape
        zt_flat = zt.reshape(num_times * batch_size, latent_dim)
        xt_flat = self.decoder(zt_flat)  # (num_times * batch_size, state_dim)

        # Reshape back
        state_dim = xt_flat.shape[-1]
        xt = xt_flat.reshape(num_times, batch_size, state_dim)

        return xt

    def predict(
        self,
        x0: torch.Tensor,
        t0: float,
        t1: float,
        num_steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict trajectory from t0 to t1.

        Args:
            x0: Initial observation of shape (batch_size, state_dim)
            t0: Initial time
            t1: Final time
            num_steps: Number of time steps

        Returns:
            Tuple of (time_points, predictions)
                - time_points: shape (num_steps,)
                - predictions: shape (num_steps, batch_size, state_dim)
        """
        device = x0.device
        t = torch.linspace(t0, t1, num_steps, device=device, dtype=x0.dtype)
        xt = self.forward(x0, t)
        return t, xt


class SimpleODEFunc(nn.Module):
    """Simple MLP-based ODE function without residual connections.

    Alternative to ODEFunc for simpler dynamics.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list = [128, 128],
        activation: nn.Module = nn.Tanh(),
        time_dependent: bool = True,
    ):
        """Initialize SimpleODEFunc.

        Args:
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            time_dependent: Whether to include time as input
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.time_dependent = time_dependent

        # Build MLP
        input_dim = latent_dim + 1 if time_dependent else latent_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt = f(t, z).

        Args:
            t: Time tensor
            z: Latent state tensor of shape (batch_size, latent_dim)

        Returns:
            Time derivative dz/dt
        """
        if self.time_dependent:
            batch_size = z.shape[0]
            t_expanded = t.expand(batch_size, 1)
            tz = torch.cat([t_expanded, z], dim=-1)
            dzdt = self.net(tz)
        else:
            dzdt = self.net(z)

        return dzdt
