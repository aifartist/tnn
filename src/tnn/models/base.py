"""
Base model utilities for Tversky Neural Networks
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class BaseClassifier(nn.Module):
    """Base class for all classifiers with common functionality"""
    
    def __init__(self):
        super().__init__()
        
    def get_feature_dim(self) -> int:
        """Return the feature dimension before the final classifier"""
        raise NotImplementedError
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier"""
        raise NotImplementedError
        
    def freeze_backbone(self):
        """Freeze the backbone parameters (everything except final layer)"""
        for name, param in self.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        """Unfreeze the backbone parameters"""
        for param in self.parameters():
            param.requires_grad = True
            
    def get_trainable_params(self) -> Dict[str, int]:
        """Get count of trainable parameters by component"""
        total = 0
        backbone = 0
        classifier = 0
        
        for name, param in self.named_parameters():
            param_count = param.numel()
            total += param_count
            
            if param.requires_grad:
                if 'fc' in name or 'classifier' in name:
                    classifier += param_count
                else:
                    backbone += param_count
                    
        return {
            'total': total,
            'backbone_trainable': backbone,
            'classifier_trainable': classifier,
            'total_trainable': backbone + classifier
        }
        
    def print_model_info(self):
        """Print model architecture and parameter information"""
        params = self.get_trainable_params()
        print(f"Model: {self.__class__.__name__}")
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable parameters: {params['total_trainable']:,}")
        print(f"  - Backbone: {params['backbone_trainable']:,}")
        print(f"  - Classifier: {params['classifier_trainable']:,}")
