#!/usr/bin/env python3
"""
XOR toy problem to test Tversky Neural Networks
Reproduces Section 3.1 from the paper with accurate hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
import argparse

# Add parent directory to path to import layers
sys.path.append(str(Path(__file__).parent.parent))

from layers import TverskyProjectionLayer
from utils import get_xor_data, plot_decision_boundary, plot_prototypes, plot_training_curves

class TverskyXORNet(nn.Module):
    """
    XOR network using Tversky projection layer
    Following the paper's architectural choices for toy problems
    """
    
    def __init__(
        self,
        hidden_dim: int = 8,
        num_prototypes: int = 4,
        alpha: float = 0.5,
        beta: float = 0.5
    ):
        super().__init__()
        
        # Simple linear transformation to hidden space
        self.hidden = nn.Linear(2, hidden_dim)
        self.activation = nn.ReLU()
        
        # Tversky projection layer with paper's hyperparameters
        self.tversky = TverskyProjectionLayer(
            input_dim=hidden_dim,
            num_prototypes=num_prototypes,
            alpha=alpha,                    # Paper uses α = 0.5
            beta=beta,                      # Paper uses β = 0.5
            theta=1e-7,                     # Small constant for stability
            intersection_reduction="product",
            difference_reduction="subtractmatch",  # Paper's preferred method
            feature_bank_init="xavier",     # Better initialization
            prototype_init="xavier",
            apply_softmax=False,
            share_feature_bank=True
        )
        
        # Final classification layer
        self.classifier = nn.Linear(num_prototypes, 1)
        
    def forward(self, x):
        h = self.activation(self.hidden(x))
        similarities = self.tversky(h)
        output = self.classifier(similarities)
        return output

def train_xor(
    n_samples: int = 1000,
    epochs: int = 500,
    lr: float = 0.01,
    hidden_dim: int = 8,
    num_prototypes: int = 4,
    alpha: float = 0.5,
    beta: float = 0.5,
    save_plots: bool = True
):
    """
    Train XOR model and visualize results
    
    Args:
        n_samples: Number of training samples
        epochs: Number of training epochs
        lr: Learning rate
        hidden_dim: Hidden dimension size
        num_prototypes: Number of prototypes in Tversky layer
        alpha: Tversky alpha parameter
        beta: Tversky beta parameter
        save_plots: Whether to save plots
    """
    print("=" * 60)
    print("TVERSKY NEURAL NETWORKS - XOR TOY PROBLEM")
    print("=" * 60)
    print(f"Hyperparameters:")
    print(f"  Samples: {n_samples}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Prototypes: {num_prototypes}")
    print(f"  Alpha (α): {alpha}")
    print(f"  Beta (β): {beta}")
    print("=" * 60)
    
    # Generate datasets
    print("Generating XOR dataset...")
    X_train, y_train = get_xor_data(n_samples, noise_std=0.1)
    X_test, y_test = get_xor_data(200, noise_std=0.1)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize model
    print("\nInitializing TverskyXORNet...")
    model = TverskyXORNet(
        hidden_dim=hidden_dim,
        num_prototypes=num_prototypes,
        alpha=alpha,
        beta=beta
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training tracking
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Periodic evaluation
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                test_pred = (torch.sigmoid(test_outputs) > 0.5).float()
                test_acc = (test_pred == y_test).float().mean()
                
                test_losses.append(test_loss.item())
                test_accuracies.append(test_acc.item())
                
                print(f'Epoch [{epoch+1:3d}/{epochs}] | '
                      f'Train Loss: {loss.item():.4f} | '
                      f'Test Loss: {test_loss.item():.4f} | '
                      f'Test Acc: {test_acc.item():.4f}')
            model.train()
    
    # Final evaluation
    print("-" * 60)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_pred = (torch.sigmoid(test_outputs) > 0.5).float()
        test_acc = (test_pred == y_test).float().mean()
        
        print(f'Final Test Accuracy: {test_acc.item():.4f}')
        
        # Compute per-class accuracy
        for class_label in [0, 1]:
            class_mask = y_test.squeeze() == class_label
            class_acc = (test_pred[class_mask] == y_test[class_mask]).float().mean()
            print(f'Class {class_label} Accuracy: {class_acc.item():.4f}')
    
    print("=" * 60)
    
    # Visualizations
    if save_plots:
        print("Generating visualizations...")
        
        # Decision boundary plot
        plot_decision_boundary(
            model, X_test, y_test,
            title="Tversky XOR Decision Boundary",
            save_path="xor_decision_boundary.png"
        )
        
        # Prototype and feature bank visualization
        prototypes = model.tversky.get_prototypes()
        feature_bank = model.tversky.get_feature_bank()
        
        plot_prototypes(
            prototypes, feature_bank,
            title="Learned Prototypes and Feature Bank",
            save_path="xor_prototypes.png"
        )
        
        # Training curves (extend test data to match training epochs)
        extended_test_losses = []
        extended_test_accs = []
        for i in range(epochs):
            if (i + 1) % 50 == 0:
                idx = ((i + 1) // 50) - 1
                if idx < len(test_losses):
                    extended_test_losses.append(test_losses[idx])
                    extended_test_accs.append(test_accuracies[idx])
            else:
                if extended_test_losses:
                    extended_test_losses.append(extended_test_losses[-1])
                    extended_test_accs.append(extended_test_accs[-1])
        
        plot_training_curves(
            train_losses, extended_test_losses,
            None, extended_test_accs,
            title="XOR Training Curves",
            save_path="xor_training_curves.png"
        )
        
        print("Plots saved successfully!")
    
    # Print learned parameters
    print("\nLearned Parameters:")
    print("-" * 30)
    print("Prototypes (Π):")
    prototypes = model.tversky.get_prototypes()
    for i, prototype in enumerate(prototypes):
        print(f"  π_{i}: {prototype.numpy()}")
    
    print("\nFeature Bank (Ω):")
    feature_bank = model.tversky.get_feature_bank()
    print(f"  Ω: {feature_bank.numpy()}")
    
    return model, test_acc.item()

def main():
    parser = argparse.ArgumentParser(description='Train Tversky XOR Network')
    parser.add_argument('--samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=8, help='Hidden dimension')
    parser.add_argument('--prototypes', type=int, default=4, help='Number of prototypes')
    parser.add_argument('--alpha', type=float, default=0.5, help='Tversky alpha parameter')
    parser.add_argument('--beta', type=float, default=0.5, help='Tversky beta parameter')
    parser.add_argument('--no-plots', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    model, accuracy = train_xor(
        n_samples=args.samples,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_prototypes=args.prototypes,
        alpha=args.alpha,
        beta=args.beta,
        save_plots=not args.no_plots
    )
    
    print(f"\n🎉 Training completed! Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
