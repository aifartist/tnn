"""
Tversky Neural Networks (TNN) - A PyTorch implementation

This package provides a faithful reproduction of the Tversky Neural Networks
paper with differentiable Tversky similarity functions.
"""

from . import layers
from . import utils
from . import models
from . import datasets
from . import training

__version__ = "1.0.0"
__author__ = "Akshath Mangudi"
__email__ = "akshathmangudi@gmail.com"

__all__ = ["layers", "utils", "models", "datasets", "training"]