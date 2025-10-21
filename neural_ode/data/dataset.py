"""Dataset for loading hyperelasticity simulation data.

Loads time-series data from an XDMF file using meshio. This implementation
expects fields to be written as point data in the XDMF time series written by
DOLFINx and does not rely on reading raw HDF5 groups.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from meshio.xdmf import TimeSeriesReader as XDMFReader
from torch.utils.data import Dataset


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

        self.field_data, self.time_steps, self.points, self.faces = self._load_data(
            time_indices
        )

        # Compute normalization statistics per field
        if self.normalize:
            self._compute_normalization_stats()
        else:
            self.field_stats = {
                field: {"mean": 0.0, "std": 1.0} for field in self.fields
            }

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
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, torch.Tensor, torch.Tensor]:
        """Load data from XDMF (meshio).

        Args:
            time_indices: Specific time step indices to load

        Returns:
            Tuple of (field_data dict, time_steps array, points tensor, faces tensor)
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

        time_steps = np.asarray(times, dtype=float)

        # Store each field separately as a dict: {field_name: (n_timesteps, n_points, n_features)}
        field_data = {}
        for field_name in self.fields:
            field_array = np.stack(
                all_steps[field_name], axis=0
            )  # (n_timesteps, n_points, n_features)

            # Subsample spatial points if requested
            if (
                self.subsample_points is not None
                and self.subsample_points < field_array.shape[1]
            ):
                rng = np.random.RandomState(42)
                indices = rng.choice(
                    field_array.shape[1], self.subsample_points, replace=False
                )
                field_array = field_array[:, indices, :]

            # Keep spatial dimensions: (n_timesteps, n_points, n_features)
            field_data[field_name] = torch.from_numpy(field_array).float()

        points_tensor = torch.from_numpy(points).float()
        faces_tensor = torch.from_numpy(faces).long()

        print(f"Loaded fields: {list(field_data.keys())}")
        for field_name, data in field_data.items():
            print(f"  {field_name}: {data.shape}")
        print(f"Number of time steps: {len(time_steps)}")
        print(f"Number of spatial points: {points.shape[0]}")
        print(f"Number of faces: {faces.shape[0]}")

        return field_data, time_steps, points_tensor, faces_tensor

    def _compute_normalization_stats(self) -> None:
        """Compute mean and std for normalization per field."""
        self.field_stats = {}
        for field_name, data in self.field_data.items():
            mean = data.mean(dim=0, keepdim=True)
            std = data.std(dim=0, keepdim=True)
            # Avoid division by zero
            std = torch.where(std > 1e-8, std, torch.ones_like(std))
            self.field_stats[field_name] = {"mean": mean, "std": std}

    def __len__(self) -> int:
        """Return number of samples (time steps)."""
        # If sequences are requested, dataset length is number of sequences
        if self.sequence_starts is not None:
            return len(self.sequence_starts)
        # Get length from first field
        first_field = next(iter(self.field_data.values()))
        return len(first_field)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Time step index

        Returns:
            Dictionary containing:
                - One key per field with its data (normalized if enabled)
                - 'time': Time value(s)
                - 'idx': Time step index
        """
        sample = {}

        if self.sequence_starts is not None:
            # idx indexes into the sequence starts
            start = self.sequence_starts[idx]
            end = start + self.seq_length

            # Add each field as separate key
            for field_name, data in self.field_data.items():
                field_seq = data[start:end]  # (seq_length, n_features)

                if self.normalize:
                    stats = self.field_stats[field_name]
                    field_seq = (field_seq - stats["mean"]) / stats["std"]

                sample[field_name] = field_seq

            time_seq = self.time_steps[start:end]
            sample["time"] = torch.tensor(time_seq, dtype=torch.float32)
            sample["idx"] = torch.tensor([start], dtype=torch.long)
        else:
            # Single time-step behavior
            for field_name, data in self.field_data.items():
                field_data = data[idx]

                if self.normalize:
                    stats = self.field_stats[field_name]
                    field_data = (field_data - stats["mean"]) / stats["std"]

                sample[field_name] = field_data

            sample["time"] = torch.tensor([self.time_steps[idx]], dtype=torch.float32)
            sample["idx"] = torch.tensor([idx], dtype=torch.long)

        return sample

    def denormalize(self, field_name: str, data: torch.Tensor) -> torch.Tensor:
        """Denormalize field data.

        Args:
            field_name: Name of the field
            data: Normalized data tensor

        Returns:
            Denormalized data tensor
        """
        if self.normalize:
            stats = self.field_stats[field_name]
            return data * stats["std"] + stats["mean"]
        return data

    def denormalize_all(
        self, sample: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Denormalize all fields in a sample.

        Args:
            sample: Dictionary with field data (and potentially 'time', 'idx')

        Returns:
            Dictionary with denormalized field data
        """
        denormalized = {}
        for key, value in sample.items():
            if key in self.fields:
                denormalized[key] = self.denormalize(key, value)
            else:
                # Keep non-field keys as-is (time, idx, etc.)
                denormalized[key] = value
        return denormalized

    @property
    def state_dim(self) -> int:
        """Return total dimension of all fields combined (n_points * n_features per field)."""
        return sum(data.shape[1] * data.shape[2] for data in self.field_data.values())

    @property
    def field_dims(self) -> Dict[str, int]:
        """Return dimensions per field as (n_points, n_features) tuples."""
        return {
            name: (data.shape[1], data.shape[2])
            for name, data in self.field_data.items()
        }
