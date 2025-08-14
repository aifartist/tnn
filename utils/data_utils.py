import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Tuple, Optional

def get_xor_data(n_samples: int = 1000, noise_std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate XOR dataset for toy problem (Section 3.1)
    
    Args:
        n_samples: Number of samples to generate
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        X: Input features of shape (n_samples, 2)
        y: Binary labels of shape (n_samples, 1)
    """
    # Generate points in [-1, 1] x [-1, 1]
    X = torch.rand(n_samples, 2) * 2 - 1
    
    # Add small amount of noise
    if noise_std > 0:
        X += torch.randn_like(X) * noise_std
    
    # XOR logic: True if signs differ
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).float().unsqueeze(1)
    
    return X, y

def get_mnist_loader(
    batch_size: int = 128,
    train: bool = True,
    download: bool = True,
    data_dir: str = "./data"
) -> DataLoader:
    """
    Get MNIST data loader following paper's experimental setup
    
    Args:
        batch_size: Batch size for training
        train: Whether to load training or test set
        download: Whether to download dataset if not present
        data_dir: Directory to store dataset
        
    Returns:
        DataLoader for MNIST
    """
    # Paper uses standard normalization for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST standard normalization
    ])
    
    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True
    )
    
    return loader

def get_nabirds_loader(
    batch_size: int = 32,
    train: bool = True,
    download: bool = True,
    data_dir: str = "./data",
    image_size: int = 224
) -> DataLoader:
    """
    Get NABirds data loader following paper's experimental setup
    
    Note: This is a placeholder implementation. The actual NABirds dataset
    would need to be downloaded separately and processed according to the paper.
    
    Args:
        batch_size: Batch size for training
        train: Whether to load training or test set
        download: Whether to download dataset if not present
        data_dir: Directory to store dataset
        image_size: Size to resize images to
        
    Returns:
        DataLoader for NABirds (placeholder using CIFAR-10)
    """
    # Paper uses ImageNet-style preprocessing for NABirds
    if train:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # Placeholder: using CIFAR-10 instead of NABirds
    # In actual implementation, you would load the NABirds dataset
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True
    )
    
    return loader

class PennTreebankDataset(Dataset):
    """
    Penn Treebank dataset for language modeling experiments
    Following the paper's experimental setup
    """
    
    def __init__(self, data_path: str, seq_length: int = 35):
        """
        Initialize Penn Treebank dataset
        
        Args:
            data_path: Path to the Penn Treebank data file
            seq_length: Sequence length for language modeling
        """
        self.seq_length = seq_length
        
        # Load and tokenize data
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple word-level tokenization
        words = text.split()
        
        # Build vocabulary
        self.vocab = sorted(set(words))
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Convert to indices
        self.data = [self.word_to_idx[word] for word in words]
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Return sequence and target (next word)
        sequence = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        target = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return sequence, target
    
    @property
    def vocab_size(self):
        return len(self.vocab)

def get_penn_treebank_loader(
    data_path: str,
    batch_size: int = 20,
    seq_length: int = 35,
    shuffle: bool = True
) -> Tuple[DataLoader, int]:
    """
    Get Penn Treebank data loader for language modeling
    
    Args:
        data_path: Path to Penn Treebank data file
        batch_size: Batch size for training
        seq_length: Sequence length for language modeling
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader and vocabulary size
    """
    dataset = PennTreebankDataset(data_path, seq_length)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )
    
    return loader, dataset.vocab_size
