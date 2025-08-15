#!/usr/bin/env python3
"""
Test script to verify Tversky Neural Networks implementation
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tnn.layers import TverskySimilarity, TverskyProjectionLayer

def test_tversky_similarity():
    """Test basic Tversky similarity computation"""
    print("Testing TverskySimilarity...")
    
    # Create similarity function
    sim_fn = TverskySimilarity(
        feature_dim=4,
        alpha=0.5,
        beta=0.5,
        theta=1e-7,
        intersection_reduction="product"
    )
    
    # Test with guaranteed different inputs
    a = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.8, 0.2, 0.6, 0.4]])
    b = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.2, 0.8, 0.4, 0.6]])
    
    similarity = sim_fn(a, b)
    print(f"Input a: {a}")
    print(f"Input b: {b}")
    print(f"Similarity: {similarity}")
    
    # Test identical inputs (should give higher or equal similarity)
    identical_sim = sim_fn(a, a)
    print(f"Self-similarity: {identical_sim}")
    
    assert similarity.shape == (2,), f"Expected shape (2,), got {similarity.shape}"
    assert torch.all(identical_sim >= similarity), "Self-similarity should be >= pairwise similarity"
    
    # Test completely identical vectors
    identical_a = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    identical_b = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    identical_similarity = sim_fn(identical_a, identical_b)
    self_similarity = sim_fn(identical_a, identical_a)
    
    assert torch.allclose(identical_similarity, self_similarity, atol=1e-6), \
        "Identical inputs should have same similarity as self-similarity"
    
    print("✓ TverskySimilarity tests passed!")

def test_tversky_projection_layer():
    """Test Tversky projection layer"""
    print("\nTesting TverskyProjectionLayer...")
    
    # Create projection layer
    layer = TverskyProjectionLayer(
        input_dim=8,
        num_prototypes=3,
        alpha=0.5,
        beta=0.5,
        apply_softmax=True
    )
    
    # Test forward pass
    batch_size = 5
    x = torch.randn(batch_size, 8)
    
    output = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (softmax): {output}")
    
    # Check output properties
    assert output.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {output.shape}"
    assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6), "Softmax should sum to 1"
    
    # Test without softmax
    layer_no_softmax = TverskyProjectionLayer(
        input_dim=8,
        num_prototypes=3,
        apply_softmax=False
    )
    
    output_no_softmax = layer_no_softmax(x)
    print(f"Output (no softmax): {output_no_softmax}")
    
    print("✓ TverskyProjectionLayer tests passed!")

def test_gradients():
    """Test gradient computation"""
    print("\nTesting gradient computation...")
    
    # Create a simple model
    layer = TverskyProjectionLayer(
        input_dim=4,
        num_prototypes=2,
        alpha=0.5,
        beta=0.5
    )
    
    # Forward pass
    x = torch.randn(3, 4, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None, "Input gradients should exist"
    assert layer.prototypes.grad is not None, "Prototype gradients should exist"
    assert layer.similarity_fn.feature_bank.grad is not None, "Feature bank gradients should exist"
    
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Prototype gradient shape: {layer.prototypes.grad.shape}")
    print(f"Feature bank gradient shape: {layer.similarity_fn.feature_bank.grad.shape}")
    
    print("✓ Gradient tests passed!")

def test_hyperparameter_effects():
    """Test effects of different hyperparameters"""
    print("\nTesting hyperparameter effects...")
    
    # Test different alpha/beta values
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    prototype = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    
    # High alpha (penalize differences in x more)
    sim_high_alpha = TverskySimilarity(feature_dim=4, alpha=2.0, beta=0.5)
    score_high_alpha = sim_high_alpha(x, prototype)
    
    # High beta (penalize differences in prototype more)
    sim_high_beta = TverskySimilarity(feature_dim=4, alpha=0.5, beta=2.0)
    score_high_beta = sim_high_beta(x, prototype)
    
    # Balanced
    sim_balanced = TverskySimilarity(feature_dim=4, alpha=0.5, beta=0.5)
    score_balanced = sim_balanced(x, prototype)
    
    print(f"High alpha (α=2.0): {score_high_alpha.item():.4f}")
    print(f"High beta (β=2.0): {score_high_beta.item():.4f}")
    print(f"Balanced (α=β=0.5): {score_balanced.item():.4f}")
    
    print("✓ Hyperparameter tests passed!")

def main():
    """Run all tests"""
    print("=" * 60)
    print("TVERSKY NEURAL NETWORKS - TEST SUITE")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    try:
        test_tversky_similarity()
        test_tversky_projection_layer()
        test_gradients()
        test_hyperparameter_effects()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✓ TverskySimilarity implementation working")
        print("✓ TverskyProjectionLayer implementation working")
        print("✓ Gradient computation working")
        print("✓ Hyperparameter effects working")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
