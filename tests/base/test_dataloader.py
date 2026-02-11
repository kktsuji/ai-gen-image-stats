"""Tests for Base DataLoader Interface

This module contains tests for the BaseDataLoader abstract class and its interface.
Tests are organized into unit tests to ensure fast execution on CPU.
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.base.dataloader import BaseDataLoader

# Test fixtures and helper classes


class DummyDataset(Dataset):
    """Minimal dataset for testing dataloaders."""

    def __init__(self, size: int = 100, input_dim: int = 10, num_classes: int = 2):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        # Return random data and label
        x = torch.randn(self.input_dim)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y


class MinimalValidDataLoader(BaseDataLoader):
    """Minimal valid dataloader implementation for testing.

    This is the simplest possible implementation that satisfies
    the BaseDataLoader interface requirements.
    """

    def __init__(
        self, batch_size: int = 32, num_workers: int = 0, train_size: int = 100
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = 20

    def get_train_loader(self) -> DataLoader:
        dataset = DummyDataset(size=self.train_size)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def get_val_loader(self) -> DataLoader:
        dataset = DummyDataset(size=self.val_size)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class DataLoaderWithoutValidation(BaseDataLoader):
    """DataLoader that returns None for validation.

    This represents experiments that don't use validation data.
    """

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    def get_train_loader(self) -> DataLoader:
        dataset = DummyDataset(size=100)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self) -> None:
        return None


class DataLoaderWithPaths(BaseDataLoader):
    """DataLoader with path attributes for testing get_config()."""

    def __init__(
        self,
        train_path: str,
        val_path: str,
        batch_size: int = 32,
        pin_memory: bool = True,
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = 4

    def get_train_loader(self) -> DataLoader:
        dataset = DummyDataset(size=100)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def get_val_loader(self) -> DataLoader:
        dataset = DummyDataset(size=20)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )


class IncompleteDataLoader(BaseDataLoader):
    """DataLoader that doesn't implement required abstract methods.

    This should fail to instantiate.
    """

    def __init__(self):
        pass

    # Missing get_train_loader() and get_val_loader() implementations


# Unit Tests


@pytest.mark.unit
class TestBaseDataLoaderInterface:
    """Test that BaseDataLoader enforces its interface requirements."""

    def test_cannot_instantiate_base_dataloader_directly(self):
        """BaseDataLoader is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseDataLoader()

    def test_cannot_instantiate_incomplete_implementation(self):
        """DataLoaders that don't implement all abstract methods cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteDataLoader()

    def test_can_instantiate_complete_implementation(self):
        """DataLoaders that implement all abstract methods can be instantiated."""
        dataloader = MinimalValidDataLoader()
        assert isinstance(dataloader, BaseDataLoader)

    def test_get_train_loader_is_abstract(self):
        """get_train_loader() method must be implemented by subclasses."""
        assert hasattr(BaseDataLoader, "get_train_loader")
        assert getattr(BaseDataLoader.get_train_loader, "__isabstractmethod__", False)

    def test_get_val_loader_is_abstract(self):
        """get_val_loader() method must be implemented by subclasses."""
        assert hasattr(BaseDataLoader, "get_val_loader")
        assert getattr(BaseDataLoader.get_val_loader, "__isabstractmethod__", False)


@pytest.mark.unit
class TestDataLoaderCreation:
    """Test dataloader creation functionality."""

    def test_create_train_loader(self):
        """Test creating a training dataloader."""
        dataloader = MinimalValidDataLoader(batch_size=16)
        train_loader = dataloader.get_train_loader()

        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 16
        assert len(train_loader.dataset) == 100

    def test_create_val_loader(self):
        """Test creating a validation dataloader."""
        dataloader = MinimalValidDataLoader(batch_size=16)
        val_loader = dataloader.get_val_loader()

        assert isinstance(val_loader, DataLoader)
        assert val_loader.batch_size == 16
        assert len(val_loader.dataset) == 20

    def test_val_loader_can_be_none(self):
        """Test that validation loader can be None for experiments without validation."""
        dataloader = DataLoaderWithoutValidation(batch_size=32)
        val_loader = dataloader.get_val_loader()

        assert val_loader is None

    def test_different_batch_sizes(self):
        """Test creating dataloaders with different batch sizes."""
        for batch_size in [8, 16, 32, 64]:
            dataloader = MinimalValidDataLoader(batch_size=batch_size)
            train_loader = dataloader.get_train_loader()
            assert train_loader.batch_size == batch_size


@pytest.mark.unit
class TestDataLoaderIteration:
    """Test iterating through dataloaders."""

    def test_iterate_train_loader(self):
        """Test iterating through training dataloader."""
        dataloader = MinimalValidDataLoader(batch_size=10, train_size=50)
        train_loader = dataloader.get_train_loader()

        batch_count = 0
        for batch_x, batch_y in train_loader:
            batch_count += 1
            assert batch_x.shape[0] <= 10  # Last batch might be smaller
            assert batch_x.shape[1] == 10  # Input dimension
            assert batch_y.shape[0] <= 10
            assert torch.all((batch_y >= 0) & (batch_y < 2))  # Valid labels

        assert batch_count == 5  # 50 samples / 10 batch_size = 5 batches

    def test_iterate_val_loader(self):
        """Test iterating through validation dataloader."""
        dataloader = MinimalValidDataLoader(batch_size=5)
        val_loader = dataloader.get_val_loader()

        batch_count = 0
        for batch_x, batch_y in val_loader:
            batch_count += 1
            assert batch_x.shape[0] <= 5
            assert batch_x.shape[1] == 10

        assert batch_count == 4  # 20 samples / 5 batch_size = 4 batches

    def test_train_loader_multiple_epochs(self):
        """Test that train loader can be iterated multiple times."""
        dataloader = MinimalValidDataLoader(batch_size=20)
        train_loader = dataloader.get_train_loader()

        # Iterate twice to simulate multiple epochs
        for epoch in range(2):
            batch_count = 0
            for batch_x, batch_y in train_loader:
                batch_count += 1
                assert batch_x.shape[0] <= 20
            assert batch_count == 5  # 100 samples / 20 batch_size


@pytest.mark.unit
class TestDataLoaderConfiguration:
    """Test dataloader configuration methods."""

    def test_get_config_basic(self):
        """Test get_config() returns correct configuration."""
        dataloader = MinimalValidDataLoader(batch_size=32, num_workers=4)
        config = dataloader.get_config()

        assert isinstance(config, dict)
        assert config["batch_size"] == 32
        assert config["num_workers"] == 4

    def test_get_config_with_paths(self):
        """Test get_config() includes path information."""
        dataloader = DataLoaderWithPaths(
            train_path="/data/train",
            val_path="/data/val",
            batch_size=16,
            pin_memory=True,
        )
        config = dataloader.get_config()

        assert config["train_path"] == "/data/train"
        assert config["val_path"] == "/data/val"
        assert config["batch_size"] == 16
        assert config["pin_memory"] is True
        assert config["num_workers"] == 4

    def test_get_config_missing_attributes(self):
        """Test get_config() works even with missing common attributes."""
        dataloader = DataLoaderWithoutValidation(batch_size=32)
        config = dataloader.get_config()

        # Should have batch_size but not all attributes
        assert "batch_size" in config
        assert config["batch_size"] == 32

    def test_repr_string(self):
        """Test __repr__() returns informative string."""
        dataloader = MinimalValidDataLoader(batch_size=32, num_workers=4)
        repr_str = repr(dataloader)

        assert "MinimalValidDataLoader" in repr_str
        assert "batch_size=32" in repr_str
        assert "num_workers=4" in repr_str


@pytest.mark.unit
class TestDataLoaderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_batch_size_larger_than_dataset(self):
        """Test dataloader when batch size is larger than dataset."""
        dataloader = MinimalValidDataLoader(batch_size=200, train_size=50)
        train_loader = dataloader.get_train_loader()

        batch_count = 0
        for batch_x, batch_y in train_loader:
            batch_count += 1
            assert batch_x.shape[0] == 50  # Should get all samples in one batch

        assert batch_count == 1  # Only one batch

    def test_batch_size_one(self):
        """Test dataloader with batch size of 1."""
        dataloader = MinimalValidDataLoader(batch_size=1, train_size=10)
        train_loader = dataloader.get_train_loader()

        batch_count = 0
        for batch_x, batch_y in train_loader:
            batch_count += 1
            assert batch_x.shape[0] == 1

        assert batch_count == 10  # 10 batches of size 1

    def test_empty_config(self):
        """Test get_config() when no common attributes exist."""

        class MinimalDataLoader(BaseDataLoader):
            def get_train_loader(self):
                return DataLoader(DummyDataset(size=10), batch_size=4)

            def get_val_loader(self):
                return None

        dataloader = MinimalDataLoader()
        config = dataloader.get_config()

        assert isinstance(config, dict)
        # Config might be empty since no common attributes are set


@pytest.mark.unit
class TestDataLoaderConsistency:
    """Test consistency and reproducibility of dataloaders."""

    def test_loader_returns_consistent_types(self):
        """Test that loaders consistently return the same types."""
        dataloader = MinimalValidDataLoader(batch_size=10)

        # Get loaders multiple times
        train_loader1 = dataloader.get_train_loader()
        train_loader2 = dataloader.get_train_loader()

        assert type(train_loader1) == type(train_loader2)
        assert isinstance(train_loader1, DataLoader)
        assert isinstance(train_loader2, DataLoader)

    def test_val_loader_none_is_consistent(self):
        """Test that validation loader consistently returns None when applicable."""
        dataloader = DataLoaderWithoutValidation(batch_size=32)

        val_loader1 = dataloader.get_val_loader()
        val_loader2 = dataloader.get_val_loader()
        val_loader3 = dataloader.get_val_loader()

        assert val_loader1 is None
        assert val_loader2 is None
        assert val_loader3 is None

    def test_loader_properties_preserved(self):
        """Test that loader properties are preserved from initialization."""
        batch_size = 16
        num_workers = 2
        dataloader = MinimalValidDataLoader(
            batch_size=batch_size, num_workers=num_workers
        )

        train_loader = dataloader.get_train_loader()
        val_loader = dataloader.get_val_loader()

        assert train_loader.batch_size == batch_size
        assert train_loader.num_workers == num_workers
        assert val_loader.batch_size == batch_size
        assert val_loader.num_workers == num_workers
