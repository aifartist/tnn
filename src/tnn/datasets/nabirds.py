"""
NABirds dataset loader for image classification experiments
Uses Deeplake for efficient streaming and loading
"""
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional
import os
from PIL import Image
import deeplake
import numpy as np

from .transforms import get_nabirds_transforms

class NABirdsDataset(Dataset):
    """
    NABirds dataset implementation using Deeplake
    Supports both local caching and streaming from Activeloop Hub
    """

    def __init__(
        self,
        split: str = 'train',
        transform=None,
        cache_dir: Optional[str] = None,
        use_local: bool = False
    ):
        """
        Initialize NABirds dataset

        Args:
            split: 'train' or 'val'
            transform: Transform pipeline to apply to images
            cache_dir: Local directory to cache the dataset
            use_local: Whether to use a local copy of the dataset
        """
        self.split = split
        self.transform = transform
        self.cache_dir = cache_dir
        self.use_local = use_local

        # Load dataset from Deeplake
        self._load_dataset()

    def _load_dataset(self):
        """Load NABirds dataset from Deeplake Hub or local cache"""
        try:
            if self.cache_dir and self.use_local:
                # Try to load from local directory structure
                local_path = os.path.join(self.cache_dir, self.split)

                if os.path.exists(local_path):
                    print(f"Loading NABirds {self.split} dataset from local directory: {local_path}")
                    self.ds = deeplake.load(local_path)
                else:
                    # Fallback to hub and cache locally
                    print(f"Local dataset not found at {local_path}")
                    print(f"Downloading from hub and caching...")
                    hub_paths = {
                        'train': 'hub://activeloop/nabirds-dataset-train',
                        'val': 'hub://activeloop/nabirds-dataset-val'
                    }
                    remote_ds = deeplake.load(hub_paths[self.split])
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    self.ds = remote_ds.copy(local_path)
            else:
                # Stream directly from hub
                hub_paths = {
                    'train': 'hub://activeloop/nabirds-dataset-train',
                    'val': 'hub://activeloop/nabirds-dataset-val'
                }

                if self.split not in hub_paths:
                    raise ValueError(f"Split '{self.split}' not supported. Use 'train' or 'val'")

                print(f"Loading NABirds {self.split} dataset from Deeplake hub...")
                self.ds = deeplake.load(hub_paths[self.split])

            print(f"NABirds {self.split} dataset loaded successfully!")
            print(f"  Total samples: {len(self.ds)}")
            print(f"  Image shape: {self.ds.images.shape}")
            print(f"  Labels shape: {self.ds.labels.shape}")

            # Use efficient pre-computed mapping for NABirds
            # NABirds typically has labels in range 1-1011, we need 0-1010
            self.label_offset = 1  # Most datasets start from 1, we need 0-indexed
            self.num_classes = 1011  # Known NABirds class count
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]

            print(f"  Number of classes: {self.num_classes}")
            print(f"  Using label offset: {self.label_offset} (subtract from original labels)")

        except Exception as e:
            print(f"Error loading NABirds dataset: {e}")
            print("Falling back to dummy data for testing...")
            self._create_dummy_data()

        except Exception as e:
            print(f"Error loading NABirds dataset: {e}")
            print("Falling back to dummy data for testing...")
            self._create_dummy_data()

    def _create_dummy_data(self):
        """Create dummy data for testing when dataset is not available"""
        print("Creating dummy NABirds data for testing...")
        # Create dummy data similar to original implementation
        self.ds = None
        self.samples = []
        self.num_classes = 400  # NABirds has 400 species
        self.class_names = [f"bird_species_{i}" for i in range(self.num_classes)]

        # Create dummy samples
        num_samples = 1000 if self.split == 'train' else 500
        for i in range(num_samples):
            class_id = i % self.num_classes
            # Create realistic bird image dimensions
            image = torch.randn(3, 224, 224)
            self.samples.append((image, class_id))
        """Create dummy data for testing when dataset is not available"""
        print("Creating dummy NABirds data for testing...")
    def __len__(self):
        """Return the number of samples in the dataset"""
        if hasattr(self, 'ds') and self.ds is not None:
            return len(self.ds)
        else:
            return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        try:
            if hasattr(self, 'ds') and self.ds is not None:
                # Using Deeplake dataset
                # Get image and label from deeplake
                image_array = self.ds.images[idx].numpy()
                label = self.ds.labels[idx].numpy()

                # Convert numpy array to PIL Image
                if image_array.dtype == np.uint8:
                    # Already in proper format
                    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                        # RGB format
                        image = Image.fromarray(image_array, mode='RGB')
                    else:
                        # Handle other formats
                        image = Image.fromarray(image_array)
                        image = image.convert('RGB')
                else:
                    # Normalize if not uint8
                    if image_array.max() <= 1.0:
                        image_array = (image_array * 255).astype(np.uint8)
                    else:
                        image_array = image_array.astype(np.uint8)
                    image = Image.fromarray(image_array, mode='RGB')

                # Apply transforms if provided
                if self.transform:
                    image = self.transform(image)

                # Ensure label is scalar and apply offset mapping
                if isinstance(label, np.ndarray):
                    label = label.item()

                # Apply simple offset mapping to normalize to [0, num_classes-1]
                if hasattr(self, 'label_offset'):
                    original_label = int(label)
                    label = original_label - self.label_offset
                    # Clamp to valid range
                    label = max(0, min(label, self.num_classes - 1))

                return image, label
            else:
                # Using dummy data
                image, label = self.samples[idx]
                if self.transform:
                    # Convert tensor to PIL for transforms, then back
                    if isinstance(image, torch.Tensor):
                        # Convert to PIL Image
                        if image.dim() == 3:  # CHW format
                            image = image.permute(1, 2, 0)  # HWC format
                        image_np = image.numpy()
                        if image_np.max() <= 1.0:
                            image_np = (image_np * 255).astype(np.uint8)
                        else:
                            image_np = image_np.astype(np.uint8)
                        image = Image.fromarray(image_np, mode='RGB')

                    image = self.transform(image)

                return image, label

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a black image and label 0 as fallback
            black_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), mode='RGB')
            if self.transform:
                black_image = self.transform(black_image)
            return black_image, 0


def get_nabirds_loaders(
    data_dir: str = './datasets/nabirds',
    batch_size: int = 64,
    frozen: bool = False,
    pretrained: bool = True,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_local: bool = True  # Default to True for downloaded datasets
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get NABirds train, validation, and test data loaders using Deeplake

    Args:
        data_dir: Directory containing NABirds dataset (default: ./datasets/nabirds)
        batch_size: Batch size for data loaders
        frozen: Whether backbone is frozen (affects augmentation)
        pretrained: Whether using pretrained weights (affects normalization)
        image_size: Target image size (224 for ResNet)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU
        use_local: Whether to use local dataset (True) or stream from hub (False)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    # Get transforms
    train_transform, val_transform = get_nabirds_transforms(frozen, pretrained, image_size)

    # Check if local dataset exists
    local_train_dir = os.path.join(data_dir, 'train')
    local_val_dir = os.path.join(data_dir, 'val')

    if use_local and os.path.exists(local_train_dir) and os.path.exists(local_val_dir):
        print(f"Using local NABirds dataset from {data_dir}")
        # Load from local directories
        train_dataset = NABirdsDataset(
            split='train',
            cache_dir=data_dir,
            use_local=True,
            transform=train_transform
        )

        val_dataset = NABirdsDataset(
            split='val',
            cache_dir=data_dir,
            use_local=True,
            transform=val_transform
        )

        test_dataset = NABirdsDataset(
            split='val',
            cache_dir=data_dir,
            use_local=True,
            transform=val_transform
        )
    else:
        if use_local:
            print(f"Local NABirds dataset not found at {data_dir}")
            print("To download the dataset, run: python download_nabirds.py")
            print("Falling back to streaming from Deeplake hub...")

        # Stream from Deeplake hub
        train_dataset = NABirdsDataset(
            split='train',
            transform=train_transform,
            cache_dir=None,
            use_local=False
        )

        val_dataset = NABirdsDataset(
            split='val',
            transform=val_transform,
            cache_dir=None,
            use_local=False
        )

        test_dataset = NABirdsDataset(
            split='val',
            transform=val_transform,
            cache_dir=None,
            use_local=False
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"NABirds Dataset loaded using Deeplake:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Classes: {getattr(train_dataset, 'num_classes', 'Unknown')}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Frozen backbone: {frozen}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Local caching: {use_local}")

    return train_loader, val_loader, test_loader
