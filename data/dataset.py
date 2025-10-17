"""Dataset for loading hyperelasticity simulation data."""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json


class HyperelasticityDataset(Dataset):
    """Dataset for hyperelasticity simulation data stored in XDMF/HDF5 format.
    
    The dataset reads displacement, velocity, stress (PK1), and energy density
    fields from the simulation output.
    """
    
    def __init__(
        self,
        data_dir: str = "results",
        fields: Optional[List[str]] = None,
        time_indices: Optional[List[int]] = None,
        normalize: bool = True,
        subsample_points: Optional[int] = None,
    ):
        """Initialize HyperelasticityDataset.
        
        Args:
            data_dir: Directory containing simulation results
            fields: List of fields to load (default: ['displacement', 'velocity'])
                Available: 'displacement', 'velocity', 'PK1', 'energy_density'
            time_indices: Specific time step indices to load (default: all)
            normalize: Whether to normalize data to zero mean and unit variance
            subsample_points: If set, randomly subsample this many spatial points
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.fields = fields or ["displacement", "velocity"]
        self.normalize = normalize
        self.subsample_points = subsample_points
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            
        # Load data from HDF5
        self.h5_path = self.data_dir / "solution.h5"
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
            
        self.data, self.time_steps = self._load_data(time_indices)
        
        # Compute normalization statistics
        if self.normalize:
            self._compute_normalization_stats()
        else:
            self.mean = 0.0
            self.std = 1.0
            
    def _load_data(
        self, 
        time_indices: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Load data from HDF5 file.
        
        Args:
            time_indices: Specific time step indices to load
            
        Returns:
            Tuple of (data tensor, time_steps array)
        """
        with h5py.File(self.h5_path, "r") as f:
            # Explore structure
            print(f"Available groups in HDF5: {list(f.keys())}")
            
            # Find all timesteps
            # The XDMF format stores fields under paths like:
            # /Mesh/<mesh_id>/displacement/<time_id>
            # We need to find the mesh group first
            
            all_data = []
            time_steps = []
            
            # Try to find the Function group structure
            # DOLFINx XDMF typically uses: /Function/<function_name>/<step>
            if "Function" in f:
                func_group = f["Function"]
                
                # Load each requested field
                field_data = []
                for field in self.fields:
                    if field in func_group:
                        field_group = func_group[field]
                        
                        # Get all time step keys (they are usually integers as strings)
                        step_keys = sorted(field_group.keys(), key=lambda x: int(x))
                        
                        if time_indices is not None:
                            step_keys = [step_keys[i] for i in time_indices if i < len(step_keys)]
                        
                        # Load data for each time step
                        field_timesteps = []
                        for step_key in step_keys:
                            data = field_group[step_key][:]
                            field_timesteps.append(data)
                            
                        field_data.append(np.stack(field_timesteps))
                        
                        # Extract time steps from first field only
                        if len(time_steps) == 0:
                            time_steps = np.array([float(k) for k in step_keys])
                            
                # Concatenate all fields along the last dimension
                # Shape: (n_timesteps, n_points, n_features)
                if len(field_data) > 0:
                    all_data = np.concatenate(field_data, axis=-1)
                else:
                    raise ValueError("No field data found")
                    
            else:
                raise ValueError("Could not find Function group in HDF5 file")
        
        # Subsample spatial points if requested
        if self.subsample_points is not None and self.subsample_points < all_data.shape[1]:
            rng = np.random.RandomState(42)
            indices = rng.choice(all_data.shape[1], self.subsample_points, replace=False)
            all_data = all_data[:, indices, :]
            
        # Flatten spatial dimensions: (n_timesteps, n_points * n_features)
        n_timesteps = all_data.shape[0]
        all_data = all_data.reshape(n_timesteps, -1)
        
        # Convert to torch tensor
        data_tensor = torch.from_numpy(all_data).float()
        
        print(f"Loaded data shape: {data_tensor.shape}")
        print(f"Number of time steps: {len(time_steps)}")
        
        return data_tensor, time_steps
    
    def _compute_normalization_stats(self) -> None:
        """Compute mean and std for normalization."""
        self.mean = self.data.mean(dim=0, keepdim=True)
        self.std = self.data.std(dim=0, keepdim=True)
        # Avoid division by zero
        self.std = torch.where(self.std > 1e-8, self.std, torch.ones_like(self.std))
        
    def __len__(self) -> int:
        """Return number of samples (time steps)."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Time step index
            
        Returns:
            Dictionary containing:
                - 'state': Flattened state vector (normalized if enabled)
                - 'time': Time value
                - 'idx': Time step index
        """
        state = self.data[idx]
        
        if self.normalize:
            state = (state - self.mean) / self.std
            
        return {
            "state": state,
            "time": torch.tensor([self.time_steps[idx]], dtype=torch.float32),
            "idx": torch.tensor([idx], dtype=torch.long),
        }
    
    def denormalize(self, state: torch.Tensor) -> torch.Tensor:
        """Denormalize state vector.
        
        Args:
            state: Normalized state tensor
            
        Returns:
            Denormalized state tensor
        """
        if self.normalize:
            return state * self.std + self.mean
        return state
    
    @property
    def state_dim(self) -> int:
        """Return dimension of state vector."""
        return self.data.shape[1]
