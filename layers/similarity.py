import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional

class TverskySimilarity(nn.Module):
    """
    Implements the Tversky similarity function from Equation 6:
    S_Ω,α,β,θ(a,b) = |a ∩ b|_Ω / (|a ∩ b|_Ω + α|a \ b|_Ω + β|b \ a|_Ω + θ)
    
    Based on the paper's hyperparameters and experimental settings.
    """
    
    def __init__(
        self,
        feature_dim: int,
        alpha: float = 0.5,  # Paper uses α = 0.5 for most experiments
        beta: float = 0.5,   # Paper uses β = 0.5 for most experiments  
        theta: float = 1e-7, # Small constant for numerical stability
        intersection_reduction: Literal["product", "mean"] = "product",
        difference_reduction: Literal["ignorematch", "subtractmatch"] = "subtractmatch",  # Paper default
        feature_bank_init: Literal["ones", "random", "xavier"] = "xavier"  # Better initialization
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.intersection_reduction = intersection_reduction
        self.difference_reduction = difference_reduction
        
        # Feature bank Ω - learnable parameters (Equation 6)
        self.feature_bank = nn.Parameter(torch.ones(feature_dim))
        self._init_feature_bank(feature_bank_init)
        
    def _init_feature_bank(self, init_type: str):
        """Initialize the feature bank Ω following paper's approach"""
        if init_type == "ones":
            nn.init.ones_(self.feature_bank)
        elif init_type == "random":
            nn.init.normal_(self.feature_bank, mean=1.0, std=0.1)
        elif init_type == "xavier":
            # Xavier/Glorot initialization scaled to positive values
            nn.init.xavier_uniform_(self.feature_bank.unsqueeze(0))
            self.feature_bank.data = torch.abs(self.feature_bank.data) + 0.1
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
    
    def _intersection(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute intersection a ∩ b with feature bank weighting
        Following the paper's definition of set intersection in feature space
        """
        if self.intersection_reduction == "product":
            # Element-wise minimum (standard set intersection)
            intersection = torch.min(a, b)
        elif self.intersection_reduction == "mean":
            intersection = (a + b) / 2
        else:
            raise ValueError(f"Unknown intersection_reduction: {self.intersection_reduction}")
        
        # Apply feature bank weighting: |·|_Ω (Equation 6)
        weighted = intersection * self.feature_bank
        return weighted.sum(dim=-1)  # Sum over feature dimension
    
    def _difference(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute difference a \ b with feature bank weighting
        Paper uses subtractmatch as default for better gradient flow
        """
        if self.difference_reduction == "ignorematch":
            # Only consider features where a > b
            diff = torch.relu(a - b)
        elif self.difference_reduction == "subtractmatch":
            # Subtract intersection from a: a - min(a,b)
            diff = a - torch.min(a, b)
        else:
            raise ValueError(f"Unknown difference_reduction: {self.difference_reduction}")
        
        # Apply feature bank weighting: |·|_Ω (Equation 6)
        weighted = diff * self.feature_bank
        return weighted.sum(dim=-1)  # Sum over feature dimension
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky similarity between tensors a and b (Equation 6)
        
        Args:
            a: Tensor of shape (..., feature_dim)
            b: Tensor of shape (..., feature_dim)
            
        Returns:
            Similarity scores of shape (...)
        """
        # Ensure positive values for set-theoretic operations
        a = torch.abs(a)
        b = torch.abs(b)
        
        # Compute components of Tversky formula
        intersection = self._intersection(a, b)  # |a ∩ b|_Ω
        diff_a_b = self._difference(a, b)        # |a \ b|_Ω  
        diff_b_a = self._difference(b, a)        # |b \ a|_Ω
        
        # Tversky similarity formula (Equation 6)
        numerator = intersection
        denominator = intersection + self.alpha * diff_a_b + self.beta * diff_b_a + self.theta
        
        # Ensure numerical stability
        similarity = numerator / torch.clamp(denominator, min=self.theta)
        return similarity
