"""Dataset for loading hyperelasticity simulation data.

Loads time-series data from an XDMF file using meshio. This implementation
expects fields to be written as point data in the XDMF time series written by
DOLFINx and does not rely on reading raw HDF5 groups.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from meshio.xdmf import TimeSeriesReader as XDMFReader
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json


def convert_xdmf_cell_to_trimesh(cell: np.ndarray, type: str):
    """Convert basix cell data to trimesh format."""
    if type == "triangle":
        # meshio triangles are (n, 3) with vertex indices
        return cell
    elif type == "hexahedron":
        # Convert hexahedra to 12 triangles
        # Hexahedron vertex ordering:
        #      3 -------- 2
        #     /|         /|
        #    7 -------- 6 |
        #    | |        | |
        #    | 0 -------|-1
        #    |/         |/
        #    4 -------- 5
        return np.vstack(
            [
                # front face
                cell[:, [4, 5, 7]],
                cell[:, [5, 6, 7]],
                # back face
                cell[:, [0, 2, 1]],
                cell[:, [0, 3, 2]],
                # left face
                cell[:, [0, 4, 3]],
                cell[:, [4, 7, 3]],
                # right face
                cell[:, [1, 2, 5]],
                cell[:, [2, 6, 5]],
                # top face
                cell[:, [3, 7, 2]],
                cell[:, [7, 6, 2]],
                # bottom face
                cell[:, [0, 1, 4]],
                cell[:, [1, 5, 4]],
            ]
        )
    else:
        raise ValueError(f"Unsupported cell type for conversion to trimesh: {type}")


class HyperelasticityDataset(Dataset):
    """Dataset for hyperelasticity simulation data stored in XDMF format.

    Uses meshio to read displacement, velocity, stress (PK1), and energy density
    fields from the simulation output. Requires an XDMF time-series file
    (e.g., 'solution.xdmf').
    """

    def __init__(
        self,
        data_dir: str = "results",
        fields: Optional[List[str]] = None,
        time_indices: Optional[List[int]] = None,
        normalize: bool = True,
        subsample_points: Optional[int] = None,
        seq_length: Optional[int] = None,
        seq_stride: int = 1,
    ):
        """Initialize HyperelasticityDataset.

        Args:
            data_dir: Directory containing simulation results
            fields: List of fields to load (default: ['displacement', 'velocity'])
                Available: 'displacement', 'velocity', 'PK1', 'energy_density'
            time_indices: Specific time step indices to load (default: all)
            normalize: Whether to normalize data to zero mean and unit variance
            subsample_points: If set, randomly subsample this many spatial points
            seq_length: If set, return sequences of this temporal length
            seq_stride: Stride between sequence starts (for sliding windows)
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.fields = fields or ["displacement", "velocity"]
        self.normalize = normalize
        self.subsample_points = subsample_points
        # Sequence sampling parameters. If seq_length is provided, the dataset
        # will return sequences of shape (seq_length, n_features) instead of
        # single time steps.
        self.seq_length = seq_length
        self.seq_stride = int(seq_stride)

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Require XDMF via meshio
        self.xdmf_path = self.data_dir / "solution.xdmf"
        if not self.xdmf_path.exists():
            raise FileNotFoundError(
                f"XDMF file not found: {self.xdmf_path}. This dataset requires 'solution.xdmf'."
            )

        self.data, self.time_steps, self.points, self.faces = self._load_data(
            time_indices
        )

        # Compute normalization statistics
        if self.normalize:
            self._compute_normalization_stats()
        else:
            self.mean = 0.0
            self.std = 1.0

        # If sequence sampling is enabled, compute valid start indices for
        # sliding window sequences.
        if self.seq_length is not None:
            n_timesteps = len(self.time_steps)
            if self.seq_length > n_timesteps:
                raise ValueError(
                    f"seq_length={self.seq_length} is larger than available time steps {n_timesteps}"
                )
            # start indices where a full sequence fits
            self.sequence_starts = list(
                range(0, n_timesteps - self.seq_length + 1, self.seq_stride)
            )
        else:
            self.sequence_starts = None

    def _load_data(
        self, time_indices: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Load data from XDMF (meshio).

        Args:
            time_indices: Specific time step indices to load

        Returns:
            Tuple of (data tensor, time_steps array)
        """
        # Only meshio XDMF path is supported
        all_steps: dict[str, list[np.ndarray]] = {}
        times: List[float] = []
        with XDMFReader(str(self.xdmf_path)) as reader:
            # Ensure points/cells can be read (also validates file)
            points, cells = reader.read_points_cells()
            faces = np.zeros((0, 3), dtype=int)
            for cell in cells:
                faces = np.vstack(
                    [faces, convert_xdmf_cell_to_trimesh(cell.data, cell.type)]
                )
            num_steps = reader.num_steps

            # Build list of step indices respecting time_indices
            step_indices = list(range(num_steps))
            if time_indices is not None:
                step_indices = [i for i in time_indices if 0 <= i < num_steps]

            # Load temporal data
            for domain in reader.domain:
                attrib = domain.attrib
                if attrib["Name"] not in self.fields:
                    continue
                if (
                    attrib["GridType"] != "Collection"
                    or attrib["CollectionType"] != "Temporal"
                ):
                    raise ValueError(
                        f"Expected XDMF domain '{attrib['Name']}' to be a temporal collection for time series data."
                    )

                for k in step_indices:
                    for element in domain[k]:
                        if element.tag == "Time" and len(times) < len(step_indices):
                            times.append(float(element.attrib["Value"]))
                        elif element.tag == "Attribute":
                            assert element.attrib["Name"] == attrib["Name"]
                            assert len(element) == 1
                            data = reader._read_data_item(element[0])
                            all_steps.setdefault(attrib["Name"], []).append(data)

        # Stack across time: (n_timesteps, n_points, n_features)
        all_data = np.concatenate(
            [np.stack(all_steps[field], axis=0) for field in self.fields],
            axis=-1,
        )
        time_steps = np.asarray(times, dtype=float)

        # Subsample spatial points if requested
        if (
            self.subsample_points is not None
            and self.subsample_points < all_data.shape[1]
        ):
            rng = np.random.RandomState(42)
            indices = rng.choice(
                all_data.shape[1], self.subsample_points, replace=False
            )
            all_data = all_data[:, indices, :]

        # Flatten spatial dimensions: (n_timesteps, n_points * n_features)
        n_timesteps = all_data.shape[0]
        all_data = all_data.reshape(n_timesteps, -1)

        # Convert to torch tensor
        data_tensor = torch.from_numpy(all_data).float()
        points_tensor = torch.from_numpy(points).float()
        faces_tensor = torch.from_numpy(faces).long()

        print(f"Loaded data shape: {data_tensor.shape}")
        print(f"Number of time steps: {len(time_steps)}")
        print(f"Number of spatial points: {points.shape[0]}")
        print(f"Number of faces: {faces.shape[0]}")

        return data_tensor, time_steps, points_tensor, faces_tensor

    def _compute_normalization_stats(self) -> None:
        """Compute mean and std for normalization."""
        self.mean = self.data.mean(dim=0, keepdim=True)
        self.std = self.data.std(dim=0, keepdim=True)
        # Avoid division by zero
        self.std = torch.where(self.std > 1e-8, self.std, torch.ones_like(self.std))

    def __len__(self) -> int:
        """Return number of samples (time steps)."""
        # If sequences are requested, dataset length is number of sequences
        if self.sequence_starts is not None:
            return len(self.sequence_starts)
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
        if self.sequence_starts is not None:
            # idx indexes into the sequence starts
            start = self.sequence_starts[idx]
            end = start + self.seq_length
            # slice along time dimension: (seq_length, n_features)
            state_seq = self.data[start:end]
            time_seq = self.time_steps[start:end]

            if self.normalize:
                state_seq = (state_seq - self.mean) / self.std

            return {
                "state": state_seq,  # (seq_length, features)
                "time": torch.tensor(time_seq, dtype=torch.float32),
                "idx": torch.tensor([start], dtype=torch.long),
            }

        # Single time-step behavior (backwards compatible)
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
