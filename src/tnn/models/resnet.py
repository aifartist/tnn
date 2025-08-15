"""
TverskyResNet implementation for image classification
Replaces the final linear layer with TverskyProjectionLayer
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any, Literal

from ..layers import TverskyProjectionLayer
from .base import BaseClassifier

# ResNet architecture configurations
RESNET_CONFIGS = {
    'resnet18': {'layers': [2, 2, 2, 2], 'feature_dim': 512},
    'resnet50': {'layers': [3, 4, 6, 3], 'feature_dim': 2048},
    'resnet101': {'layers': [3, 4, 23, 3], 'feature_dim': 2048},
    'resnet152': {'layers': [3, 8, 36, 3], 'feature_dim': 2048},
}

class TverskyResNet(BaseClassifier):
    """
    ResNet with Tversky projection layer as final classifier
    
    Based on the paper's experimental setup for image classification tasks.
    Supports all ResNet variants (18, 50, 101, 152) with configurable
    pretrained weights and freezing options.
    """
    
    def __init__(
        self,
        architecture: Literal['resnet18', 'resnet50', 'resnet101', 'resnet152'] = 'resnet18',
        num_classes: int = 10,
        pretrained: bool = True,
        frozen: bool = False,
        use_tversky: bool = True,
        # Tversky layer parameters (from paper and XOR success)
        num_prototypes: int = 8,
        alpha: float = 0.5,
        beta: float = 0.5,
        theta: float = 1e-7,
        intersection_reduction: Literal["product", "mean"] = "product",
        difference_reduction: Literal["ignorematch", "subtractmatch"] = "subtractmatch",
        feature_bank_init: Literal["ones", "random", "xavier"] = "xavier",
        prototype_init: Literal["random", "uniform", "normal", "xavier"] = "xavier",
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.frozen = frozen
        self.use_tversky = use_tversky
        
        # Validate architecture
        if architecture not in RESNET_CONFIGS:
            raise ValueError(f"Unsupported architecture: {architecture}. "
                           f"Supported: {list(RESNET_CONFIGS.keys())}")
        
        self.config = RESNET_CONFIGS[architecture]
        
        # Load backbone model
        self.backbone = self._create_backbone()
        
        # Create final classifier
        feature_dim = self.config['feature_dim']
        
        if use_tversky:
            # Use Tversky projection layer (following XOR success pattern)
            self.fc = TverskyProjectionLayer(
                input_dim=feature_dim,
                num_prototypes=num_prototypes,
                alpha=alpha,
                beta=beta,
                theta=1e-5,  # Slightly larger for stability like XOR
                intersection_reduction=intersection_reduction,
                difference_reduction=difference_reduction,
                feature_bank_init=feature_bank_init,
                prototype_init=prototype_init,
                apply_softmax=False,  # Key change: no softmax in Tversky layer
                share_feature_bank=True,
                temperature=temperature
            )
            # Single linear layer to map prototypes to classes
            self.prototype_to_class = nn.Linear(num_prototypes, num_classes)
            
            # Better initialization for the classification layer
            nn.init.xavier_uniform_(self.prototype_to_class.weight)
            nn.init.zeros_(self.prototype_to_class.bias)
        else:
            # Use standard linear layer (baseline)
            self.fc = nn.Linear(feature_dim, num_classes)
            self.prototype_to_class = None
        
        # Apply freezing if requested
        if frozen:
            self.freeze_backbone()
            
    def _create_backbone(self) -> nn.Module:
        """Create the ResNet backbone without the final fc layer"""
        # Load the appropriate ResNet model
        if self.architecture == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
        elif self.architecture == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
        elif self.architecture == 'resnet101':
            model = models.resnet101(pretrained=self.pretrained)
        elif self.architecture == 'resnet152':
            model = models.resnet152(pretrained=self.pretrained)
        
        # Remove the final fc layer
        backbone = nn.Sequential(*list(model.children())[:-1])
        return backbone
        
    def get_feature_dim(self) -> int:
        """Return the feature dimension before the final classifier"""
        return self.config['feature_dim']
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier"""
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        # Extract features using backbone
        features = self.get_features(x)
        
        # Apply final classifier
        if self.use_tversky:
            # Tversky projection layer outputs prototype similarities
            prototype_scores = self.fc(features)
            # Map to class probabilities
            class_logits = self.prototype_to_class(prototype_scores)
            return class_logits
        else:
            # Standard linear layer
            return self.fc(features)
            
    def get_prototype_similarities(self, x: torch.Tensor) -> torch.Tensor:
        """Get prototype similarities (only available for Tversky models)"""
        if not self.use_tversky:
            raise ValueError("Prototype similarities only available for Tversky models")
            
        features = self.get_features(x)
        prototype_scores = self.fc(features)
        return prototype_scores
        
    def get_prototypes(self) -> Optional[torch.Tensor]:
        """Get learned prototypes (only available for Tversky models)"""
        if not self.use_tversky:
            return None
        return self.fc.get_prototypes()
        
    def get_feature_bank(self) -> Optional[torch.Tensor]:
        """Get learned feature bank (only available for Tversky models)"""
        if not self.use_tversky:
            return None
        return self.fc.get_feature_bank()


def get_resnet_model(
    architecture: str = 'resnet18',
    num_classes: int = 10,
    pretrained: bool = True,
    frozen: bool = False,
    use_tversky: bool = True,
    **tversky_kwargs
) -> TverskyResNet:
    """
    Factory function to create TverskyResNet models
    
    Args:
        architecture: ResNet architecture ('resnet18', 'resnet50', 'resnet101', 'resnet152')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained ImageNet weights
        frozen: Whether to freeze backbone parameters
        use_tversky: Whether to use Tversky layer (False for baseline)
        **tversky_kwargs: Additional arguments for TverskyProjectionLayer
        
    Returns:
        Configured TverskyResNet model
    """
    
    # Default Tversky parameters from paper and XOR experiments
    default_tversky_params = {
        'num_prototypes': 8,
        'alpha': 0.5,
        'beta': 0.5,
        'theta': 1e-7,
        'intersection_reduction': 'product',
        'difference_reduction': 'subtractmatch',
        'feature_bank_init': 'xavier',
        'prototype_init': 'xavier',
        'temperature': 1.0
    }
    
    # Update with user provided parameters
    default_tversky_params.update(tversky_kwargs)
    
    model = TverskyResNet(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        frozen=frozen,
        use_tversky=use_tversky,
        **default_tversky_params
    )
    
    return model
