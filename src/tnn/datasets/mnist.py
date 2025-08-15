"""
MNIST dataset loader for image classification experiments
Handles ResNet-compatible preprocessing (grayscale -> RGB, resize to 224)
"""
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional

from .transforms import get_mnist_transforms

def get_mnist_loaders(
    data_dir: str = './data',
    batch_size: int = 64,
    frozen: bool = False,
    pretrained: bool = True,
    image_size: int = 224,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get MNIST train, validation, and test data loaders
    
    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for data loaders
        frozen: Whether backbone is frozen (affects augmentation)
        pretrained: Whether using pretrained weights (affects normalization)
        image_size: Target image size (224 for ResNet)
        val_split: Fraction of training data to use for validation
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Get transforms
    train_transform, val_transform = get_mnist_transforms(frozen, pretrained, image_size)
    
    # Convert grayscale to RGB for ResNet compatibility
    class GrayscaleToRGB:
        def __call__(self, x):
            if x.size(0) == 1:  # Grayscale
                return x.repeat(3, 1, 1)  # Convert to 3-channel
            return x
    
    # Add RGB conversion to transforms
    train_transform.transforms.insert(-1, GrayscaleToRGB())  # Before normalization
    val_transform.transforms.insert(-1, GrayscaleToRGB())
    
    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=val_transform,
        download=True
    )
    
    # Split training set into train/validation
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible splits
    )
    
    # Update validation subset to use validation transforms
    val_subset.dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=val_transform,
        download=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_subset,
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
    
    print(f"MNIST Dataset loaded:")
    print(f"  Train: {len(train_subset)} samples")
    print(f"  Val: {len(val_subset)} samples")  
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Frozen backbone: {frozen}")
    print(f"  Pretrained: {pretrained}")
    
    return train_loader, val_loader, test_loader
