import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional

class TverskySimilarity(nn.Module):
    """
    Implements the Tversky similarity function from Equation 6:
    S_Ω,α,β,θ(a,b) = |a ∩ b|_Ω / (|a ∩ b|_Ω + α|a \\ b|_Ω + β|b \\ a|_Ω + θ)
    
    Based on the paper's hyperparameters and experimental settings.
    """
    
    def __init__(
        self,
        feature_dim: int,
        alpha: float = 0.5,  # Paper uses α = 0.5 for most experiments
        beta: float = 0.5,   # Paper uses β = 0.5 for most experiments  
        theta: float = 1e-7, # Small constant for numerical stability
        intersection_reduction: Literal["product", "mean", "min", "max", "gmean", "softmin"] = "product",
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
            # Ensure positive values
            self.feature_bank.data = torch.abs(self.feature_bank.data) + 0.1
        elif init_type == "xavier":
            # Xavier/Glorot initialization scaled to positive values
            nn.init.xavier_uniform_(self.feature_bank.unsqueeze(0))
            self.feature_bank.data = torch.abs(self.feature_bank.data) + 0.1
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
            
        # Additional safety: ensure feature bank is always positive and bounded
        with torch.no_grad():
            self.feature_bank.data = torch.clamp(self.feature_bank.data, min=0.1, max=10.0)
    
    def _intersection(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute intersection a ∩ b with feature bank weighting
        Following the paper's definition of set intersection in feature space
        """
        intersection = TverskySimilarity.apply_intersection_reduction(self.intersection_reduction, a, b)
        # Apply feature bank weighting: |·|_Ω (Equation 6)
        weighted = intersection * self.feature_bank
        return weighted.sum(dim=-1)  # Sum over feature dimension
    
    def _difference(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute difference a \\ b with feature bank weighting
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
        
        # Add small epsilon to prevent numerical issues
        eps = 1e-8
        a = a + eps
        b = b + eps
        
        # Ensure feature bank is positive and well-conditioned
        feature_bank_stable = torch.clamp(torch.abs(self.feature_bank), min=0.1, max=5.0)
        
        # Compute components of Tversky formula
        intersection = self._intersection_stable(a, b, feature_bank_stable)  # |a ∩ b|_Ω
        diff_a_b = self._difference_stable(a, b, feature_bank_stable)        # |a \\ b|_Ω  
        diff_b_a = self._difference_stable(b, a, feature_bank_stable)        # |b \\ a|_Ω
        
        # Tversky similarity formula (Equation 6)
        numerator = intersection
        denominator = intersection + self.alpha * diff_a_b + self.beta * diff_b_a + self.theta
        
        # Ensure numerical stability with larger minimum threshold
        min_denom = max(self.theta, 1e-5)
        denominator = torch.clamp(denominator, min=min_denom)
        
        # Additional safety check for NaN/Inf
        similarity = numerator / denominator
        similarity = torch.clamp(similarity, min=0.0, max=1.0)  # Tversky similarity should be in [0,1]
        
        # Replace any remaining NaN with small positive value
        similarity = torch.where(torch.isnan(similarity), torch.tensor(eps, device=similarity.device), similarity)
        
        return similarity
    
    def _intersection_stable(self, a: torch.Tensor, b: torch.Tensor, feature_bank: torch.Tensor) -> torch.Tensor:
        """Stable intersection computation"""
        intersection = TverskySimilarity.apply_intersection_reduction(self.intersection_reduction, a, b)
        weighted = intersection * feature_bank
        return weighted.sum(dim=-1)
    
    def _difference_stable(self, a: torch.Tensor, b: torch.Tensor, feature_bank: torch.Tensor) -> torch.Tensor:
        """Stable difference computation"""
        if self.difference_reduction == "ignorematch":
            diff = torch.relu(a - b)
        elif self.difference_reduction == "subtractmatch":
            diff = a - torch.min(a, b)
        else:
            raise ValueError(f"Unknown difference_reduction: {self.difference_reduction}")
        
        weighted = diff * feature_bank
        return weighted.sum(dim=-1)
    
    SOFTMIN_ALPHA = -1.0

    @staticmethod
    def apply_intersection_reduction(intersection_reduction: Literal["product", "mean", "min", "max", "gmean", "softmin"], a: torch.Tensor, b: torch.Tensor):
        match intersection_reduction:
            case "product":
                return a * b
            case "mean":
                return (a + b) / 2
            case "min":
                return torch.min(a, b)
            case "max":
                return torch.max(a, b)
            case "gmean":
                log_a = torch.log(a)
                log_b = torch.log(b)
                return torch.exp((log_a + log_b) / 2)
            case "softmin":
                """For Softmin, LogSumExp with a negative SOFTMIN_ALPHA is used"""
                exp_a = torch.exp(TverskySimilarity.SOFTMIN_ALPHA * a)
                exp_b = torch.exp(TverskySimilarity.SOFTMIN_ALPHA * b)
                return (1.0 / TverskySimilarity.SOFTMIN_ALPHA) * torch.log(exp_a + exp_b)
            case _:
                raise ValueError(f"Unknown intersection_reduction: {intersection_reduction}")
        

