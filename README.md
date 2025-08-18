# Tversky Neural Networks (TNN)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

A PyTorch implementation of **Tversky Neural Networks (TNNs)**, a novel architecture that replaces traditional linear classification layers with Tversky similarity-based projection layers. This implementation faithfully reproduces the key concepts from the original paper and provides optimized, production-ready models for both research and practical applications.

## 🚀 What are Tversky Neural Networks?

Tversky Neural Networks introduce a fundamentally different approach to neural network classification by leveraging **Tversky similarity functions** instead of traditional dot-product operations. The key innovation is the **Tversky Projection Layer**, which:

- **Replaces linear layers** with learnable prototype-based similarity computations
- **Uses asymmetric similarity** through Tversky index (α, β parameters)
- **Provides interpretable representations** through learned prototypes
- **Maintains competitive accuracy** while offering explainable decision boundaries

### Core Mathematical Foundation

The Tversky Projection Layer computes similarities between input features and learned prototypes using:

```
S_Ω,α,β,θ(x, π_k) = |x ∩ π_k|_Ω / (|x ∩ π_k|_Ω + α|x \ π_k|_Ω + β|π_k \ x|_Ω + θ)
```

Where:
- `x` is the input feature vector
- `π_k` are learned prototypes  
- `Ω` is a learned feature bank
- `α, β` control asymmetric similarity weighting
- `θ` provides numerical stability

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install tnn
```

### From Source

```bash
git clone https://github.com/akshathmangudi/tnn.git
cd tnn
pip install -e .
```

### Dependencies

- Python 3.10+
- PyTorch 2.0+
- torchvision 0.15+
- numpy
- scikit-learn
- tqdm
- pillow

## 🎯 Quick Start

### Basic Usage

```python
import torch
from tnn.models import get_resnet_model
from tnn.datasets import get_mnist_loaders

# Create a TverskyResNet model
model = get_resnet_model(
    architecture='resnet18',
    num_classes=10,
    use_tversky=True,
    num_prototypes=8,
    alpha=0.5,
    beta=0.5
)

# Load MNIST dataset
train_loader, val_loader, test_loader = get_mnist_loaders(
    data_dir='./data',
    batch_size=64
)

# Use the model
x = torch.randn(32, 3, 224, 224)  # Batch of images
outputs = model(x)  # Shape: (32, 10)
```

### XOR Toy Problem

Demonstrate TNN capabilities on the classic XOR problem:

```python
from tnn.models.xor import TverskyXORNet
import torch

# Create XOR model
model = TverskyXORNet(
    hidden_dim=8,
    num_prototypes=4,
    alpha=0.5,
    beta=0.5
)

# XOR data
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([0, 1, 1, 0])

# Forward pass
predictions = model(x)
```

## 🏃‍♂️ Training Models

### MNIST Classification

Train a TverskyResNet on MNIST:

```bash
# Train with Tversky layer (recommended)
python train_resnet.py --dataset mnist --architecture resnet18 --epochs 50 --lr 0.01

# Train baseline (linear layer)
python train_resnet.py --dataset mnist --architecture resnet18 --use-linear --epochs 50 --lr 0.01

# Quick test (2 epochs)
python train_resnet.py --dataset mnist --epochs 2 --lr 0.01
```

### XOR Toy Problem

```bash
python train_xor.py
```

### Advanced Training Options

TO BE UPDATED

## 📊 Results

Our implementation achieves strong performance across different tasks:

### MNIST Classification Results

| Configuration | Architecture | Classifier | Val Accuracy | Train Accuracy | Training Time |
|---------------|-------------|------------|--------------|----------------|---------------|
| **Optimized TNN** | ResNet18 | Tversky (8 prototypes) | **98.88%** | **98.81%** | ~32 min (2 epochs) |
| Baseline | ResNet18 | Linear | - | - | - |

**Key Training Metrics:**
- **Epoch 1:** Training Acc: 89.81%
- **Epoch 2:** Training Acc: 98.81%, Validation Acc: 98.88%
- **Model Size:** 11.18M parameters (4,608 in Tversky classifier)
- **Convergence:** Fast and stable with proper hyperparameters

### XOR Toy Problem Results

| Metric | Value |
|--------|-------|
| **Final Test Accuracy** | **93.00%** |
| Class 0 Accuracy | 95.40% |
| Class 1 Accuracy | 91.15% |
| Training Epochs | 500 |
| Convergence | Smooth, interpretable decision boundary |

**Visual Results:**
- Clear non-linear decision boundary
- Interpretable learned prototypes
- Smooth training curves

## 🔬 Key Features

### ✅ What Works Well

1. **Fast Convergence**: With proper hyperparameters (lr=0.01), TNNs converge quickly
2. **High Accuracy**: Achieves 98.88% validation accuracy on MNIST
3. **Interpretability**: Learned prototypes provide insight into model decisions
4. **Flexibility**: Support for multiple ResNet architectures
5. **Stability**: Robust training with mixed precision and proper initialization

### 🏗️ Architecture Highlights

- **Modular Design**: Easy to swap Tversky layers for linear layers
- **Multiple Architectures**: ResNet18/50/101/152 support
- **Pretrained Weights**: ImageNet initialization available
- **Mixed Precision**: Automatic mixed precision training
- **Comprehensive Logging**: Detailed metrics and checkpointing

### 🎛️ Configurable Hyperparameters

```python
# Tversky similarity parameters
alpha: float = 0.5              # Controls importance of false positives
beta: float = 0.5               # Controls importance of false negatives  
num_prototypes: int = 8         # Number of learned prototypes
theta: float = 1e-7             # Numerical stability constant

# Architecture options
intersection_reduction = "product"        # or "mean"
difference_reduction = "subtractmatch"    # or "ignorematch"
feature_bank_init = "xavier"             # Feature bank initialization
prototype_init = "xavier"                # Prototype initialization
```

## 🚧 Current Limitations & Future Work

### Known Issues Resolved ✅

- **Double Classification Layer**: Fixed architecture that was causing convergence issues
- **Softmax Placement**: Corrected `apply_softmax=False` in Tversky layer
- **Learning Rate**: Optimized default learning rate from 0.001 → 0.01
- **Initialization**: Improved prototype and feature bank initialization

### Future Enhancements 🔮

1. **Extended Datasets**: Support for CIFAR-10/100, ImageNet
2. **Additional Architectures**: Vision Transformers, EfficientNets
3. **Advanced Features**: 
   - Prototype visualization tools
   - Attention mechanisms
   - Multi-modal support
4. **Optimization**: 
   - Further convergence improvements
   - Memory optimization for large models
5. **Research Extensions**:
   - Adaptive α, β parameters
   - Hierarchical prototypes
   - Ensemble methods

## 📈 Performance Optimizations Applied

Our implementation includes several key optimizations discovered during development:

1. **Architectural Fixes**:
   - Removed double classification layer causing gradient flow issues
   - Set `apply_softmax=False` in Tversky layer for better optimization
   - Improved linear layer initialization with Xavier uniform

2. **Training Optimizations**:
   - Increased learning rate to 0.01 for faster convergence
   - Mixed precision training for memory efficiency
   - Cosine annealing scheduler for better convergence

3. **Numerical Stability**:
   - Proper theta parameter (1e-7) for numerical stability
   - Xavier initialization for all learnable parameters
   - Gradient clipping and proper loss scaling

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- Additional dataset implementations
- New architecture support  
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## To add: 
- [ ] Include GPT-2 implementation and benchmarks. 
- [ ] Run ResNet18 benchmarks on NABirds Dataset. 
- [ ] Add benchmarks for different datasets for different weight distributions. 
- [X] Unify training configuration instead of keeping several training files for different models. 
- [ ] Include type checking and other software development process standards to maintain robustness. 

## 📝 Citation

If you use this implementation in your research, please cite:

```bibtex
@software{tnn_pytorch,
  author = {Akshath Mangudi},
  title = {TNN: A PyTorch Implementation of Tversky Neural Networks},
  year = {2025},
  url = {https://github.com/akshathmangudi/tnn}
}
```

For the original Tversky Neural Networks paper, please cite:
```bibtex
@article{tversky_neural_networks,
  title={Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity},
  author={[Moussa Koulako Bala Doumbouya, Dan Jurafsky, Christopher D. Manning]},
  journal={[NeurIPS]},
  year={[2025]},
  url={[https://arxiv.org/abs/2506.11035]}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original Tversky Neural Networks paper authors
- PyTorch team for the excellent deep learning framework
- torchvision for pretrained models and datasets

---

**Built with ❤️ and PyTorch** | **Ready for production use** | **Optimized for research**
