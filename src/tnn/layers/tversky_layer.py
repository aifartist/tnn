import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Union
from .similarity import TverskySimilarity

class TverskyProjectionLayer(nn.Module):
    """
    Implements the Tversky Projection Layer from Equation 7:
    y_k = S_Ω,α,β,θ(x, π_k)
    
    This layer computes similarity between input and K prototypes.
    Hyperparameters follow the paper's experimental settings.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_prototypes: int,
        alpha: float = 0.5,    # Paper's default α
        beta: float = 0.5,     # Paper's default β  
        theta: float = 1e-7,   # Small constant for stability
        intersection_reduction: Literal["product", "mean", "min", "max", "gmean", "softmin"] = "product",
        difference_reduction: Literal["ignorematch", "subtractmatch"] = "subtractmatch",
        feature_bank_init: Literal["ones", "random", "xavier"] = "xavier",
        prototype_init: Literal["random", "uniform", "normal", "xavier"] = "xavier",
        apply_softmax: bool = False,
        share_feature_bank: bool = True,  # Paper typically shares feature banks
        temperature: float = 1.0  # For temperature scaling in softmax
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_prototypes = num_prototypes
        self.apply_softmax = apply_softmax
        self.share_feature_bank = share_feature_bank
        self.temperature = temperature
        
        # Prototype bank Π - learnable parameters (Equation 7)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        self._init_prototypes(prototype_init)
        
        # Tversky similarity function(s)
        if share_feature_bank:
            # Single shared feature bank for all prototypes (paper's typical setup)
            self.similarity_fn = TverskySimilarity(
                feature_dim=input_dim,
                alpha=alpha,
                beta=beta,
                theta=theta,
                intersection_reduction=intersection_reduction,
                difference_reduction=difference_reduction,
                feature_bank_init=feature_bank_init
            )
        else:
            # Separate feature bank for each prototype
            self.similarity_fns = nn.ModuleList([
                TverskySimilarity(
                    feature_dim=input_dim,
                    alpha=alpha,
                    beta=beta,
                    theta=theta,
                    intersection_reduction=intersection_reduction,
                    difference_reduction=difference_reduction,
                    feature_bank_init=feature_bank_init
                ) for _ in range(num_prototypes)
            ])
    
    def _init_prototypes(self, init_type: str):
        """Initialize the prototype bank Π following paper's approach"""
        if init_type == "random":
            nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        elif init_type == "uniform":
            nn.init.uniform_(self.prototypes, a=-1.0, b=1.0)
        elif init_type == "normal":
            nn.init.normal_(self.prototypes, mean=0.0, std=0.1)
        elif init_type == "xavier":
            nn.init.xavier_uniform_(self.prototypes)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores between input and all prototypes (Equation 7)
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Similarity scores of shape (batch_size, num_prototypes)
        """
        batch_size = x.shape[0]
        similarities = torch.zeros(batch_size, self.num_prototypes, device=x.device, dtype=x.dtype)
        
        if self.share_feature_bank:
            # Use shared feature bank for all prototypes (more efficient)
            for k in range(self.num_prototypes):
                prototype_k = self.prototypes[k].unsqueeze(0).expand(batch_size, -1)
                similarities[:, k] = self.similarity_fn(x, prototype_k)
        else:
            # Use separate feature bank for each prototype
            for k in range(self.num_prototypes):
                prototype_k = self.prototypes[k].unsqueeze(0).expand(batch_size, -1)
                similarities[:, k] = self.similarity_fns[k](x, prototype_k)
        
        # Apply temperature scaling if using softmax
        if self.apply_softmax:
            similarities = F.softmax(similarities / self.temperature, dim=-1)
            
        return similarities
    
    def get_prototypes(self) -> torch.Tensor:
        """Return the current prototype bank Π"""
        return self.prototypes.detach().clone()
    
    def get_feature_bank(self) -> Union[torch.Tensor, list]:
        """Return the current feature bank(s) Ω"""
        if self.share_feature_bank:
            return self.similarity_fn.feature_bank.detach().clone()
        else:
            return [sim_fn.feature_bank.detach().clone() 
                   for sim_fn in self.similarity_fns]
    
    def set_temperature(self, temperature: float):
        """Update temperature for softmax scaling"""
        self.temperature = temperature
        
    def get_prototype_distances(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get L2 distances to prototypes for analysis
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Distances of shape (batch_size, num_prototypes)
        """
        batch_size = x.shape[0]
        distances = torch.zeros(batch_size, self.num_prototypes, device=x.device)
        
        for k in range(self.num_prototypes):
            prototype_k = self.prototypes[k].unsqueeze(0).expand(batch_size, -1)
            distances[:, k] = torch.norm(x - prototype_k, p=2, dim=-1)
            
        return distances
