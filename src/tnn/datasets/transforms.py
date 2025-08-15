"""
Data transforms for image classification experiments
Handles both frozen and unfrozen backbone scenarios
"""
import torchvision.transforms as transforms
from typing import Tuple

# ImageNet statistics for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(
    dataset: str = 'mnist',
    frozen: bool = False,
    pretrained: bool = True,
    image_size: int = 224
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and validation transforms for different scenarios
    
    Args:
        dataset: Dataset name ('mnist' or 'nabirds')
        frozen: Whether backbone is frozen (affects augmentation strategy)
        pretrained: Whether using pretrained weights (affects normalization)
        image_size: Target image size for ResNet (default: 224)
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    
    # Choose normalization statistics
    if pretrained:
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    else:
        # Dataset-specific normalization
        if dataset == 'mnist':
            normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        else:  # nabirds or other
            normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    # Base transforms for all datasets
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ]
    
    if frozen:
        # Minimal augmentation for frozen backbone
        train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),  # Light augmentation
            transforms.ToTensor(),
            normalize
        ])
    else:
        # More aggressive augmentation for unfrozen backbone
        if dataset == 'mnist':
            # MNIST-specific augmentation
            train_transforms = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            # NABirds and other natural image datasets
            train_transforms = transforms.Compose([
                transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),  # 256 for 224
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize
            ])
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose(base_transforms)
    
    return train_transforms, val_transforms


def get_mnist_transforms(
    frozen: bool = False,
    pretrained: bool = True,
    image_size: int = 224
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Convenience function for MNIST transforms"""
    return get_transforms('mnist', frozen, pretrained, image_size)


def get_nabirds_transforms(
    frozen: bool = False,
    pretrained: bool = True,
    image_size: int = 224
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Convenience function for NABirds transforms"""
    return get_transforms('nabirds', frozen, pretrained, image_size)
