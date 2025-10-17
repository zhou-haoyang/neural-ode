"""LightningDataModule for hyperelasticity simulation data."""

import lightning as L
from torch.utils.data import DataLoader, random_split
from typing import Optional, List

from .dataset import HyperelasticityDataset


class HyperelasticityDataModule(L.LightningDataModule):
    """DataModule for hyperelasticity simulation data.
    
    Handles train/val/test split and creates dataloaders.
    """
    
    def __init__(
        self,
        data_dir: str = "results",
        fields: Optional[List[str]] = None,
        batch_size: int = 32,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        normalize: bool = True,
        subsample_points: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """Initialize HyperelasticityDataModule.
        
        Args:
            data_dir: Directory containing simulation results
            fields: List of fields to load
            batch_size: Batch size for dataloaders
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            normalize: Whether to normalize data
            subsample_points: If set, randomly subsample this many spatial points
            num_workers: Number of workers for dataloaders
            pin_memory: Whether to pin memory for dataloaders
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.fields = fields or ["displacement", "velocity"]
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.normalize = normalize
        self.subsample_points = subsample_points
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Train/val/test splits must sum to 1.0"
            
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training, validation, and testing.
        
        Args:
            stage: Stage ('fit', 'validate', 'test', or 'predict')
        """
        if self.dataset is None:
            # Load full dataset
            self.dataset = HyperelasticityDataset(
                data_dir=self.data_dir,
                fields=self.fields,
                normalize=self.normalize,
                subsample_points=self.subsample_points,
            )
            
            # Split into train/val/test
            total_size = len(self.dataset)
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size
            
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )
            
            print(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    @property
    def state_dim(self) -> Optional[int]:
        """Return dimension of state vector."""
        if self.dataset is not None:
            return self.dataset.state_dim
        return None


# Import torch for random_split
import torch
