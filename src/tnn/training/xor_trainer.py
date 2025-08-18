"""
XOR trainer extracted from train_xor.py for unified pipeline
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple
from tqdm import tqdm

from ..layers import TverskyProjectionLayer
from ..utils import get_xor_data, plot_decision_boundary, plot_prototypes, plot_training_curves
from .config import UnifiedConfig
from typing import Literal, Optional

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
        beta: float = 0.5,
        intersection_reduction: Literal["product", "mean", "min", "max", "gmean", "softmin"] = "product",
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
            theta=1e-5,                     # Slightly larger for stability
            intersection_reduction=intersection_reduction,
            difference_reduction="subtractmatch",  # Paper's preferred method
            feature_bank_init="xavier",     # Better initialization
            prototype_init="xavier",
            apply_softmax=False,
            share_feature_bank=True,
            temperature=1.0
        )
        
        # Final classification layer
        self.classifier = nn.Linear(num_prototypes, 1)
        
    def forward(self, x):
        h = self.activation(self.hidden(x))
        similarities = self.tversky(h)
        output = self.classifier(similarities)
        return output

class XORTrainer:
    """
    Trainer for XOR toy problem experiments
    """
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.xor_config = config.xor_config
        self.device = self._setup_device()
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize model, data, optimizer
        self.model = self._create_model()
        self.train_data, self.test_data = self._load_data()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training tracking
        self.train_losses = []
        self.test_losses = []
        self.test_accuracies = []
    
    def _setup_device(self) -> torch.device:
        """Setup device for training"""
        if self.config.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config.device)
        return device
    
    def _create_model(self) -> TverskyXORNet:
        """Create XOR model"""
        model = TverskyXORNet(
            hidden_dim=self.xor_config.hidden_dim,
            num_prototypes=self.config.tversky.num_prototypes,
            alpha=self.config.tversky.alpha,
            beta=self.config.tversky.beta,
            intersection_reduction=self.config.tversky.intersection_reduction
        )
        return model.to(self.device)
    
    def _load_data(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Load XOR datasets"""
        X_train, y_train = get_xor_data(
            self.xor_config.n_samples, 
            noise_std=self.xor_config.noise_std
        )
        X_test, y_test = get_xor_data(
            self.xor_config.test_samples, 
            noise_std=self.xor_config.noise_std
        )
        
        # Move to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        return (X_train, y_train), (X_test, y_test)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        if self.config.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=20
            )
        elif self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.epochs // 3, gamma=0.1
            )
        else:
            return None
    
    def train(self) -> Dict[str, Any]:
        """
        Train the XOR model
        
        Returns:
            Dictionary containing training results and metrics
        """
        print("=" * 60)
        print("TVERSKY NEURAL NETWORKS - XOR TOY PROBLEM")
        print("=" * 60)
        print(f"Hyperparameters:")
        print(f"  Samples: {self.xor_config.n_samples}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Hidden dim: {self.xor_config.hidden_dim}")
        print(f"  Prototypes: {self.config.tversky.num_prototypes}")
        print(f"  Alpha (α): {self.config.tversky.alpha}")
        print(f"  Beta (β): {self.config.tversky.beta}")
        print("=" * 60)
        
        X_train, y_train = self.train_data
        X_test, y_test = self.test_data
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"\nTraining for {self.config.epochs} epochs...")
        print("-" * 60)
        
        # Training loop
        self.model.train()
        best_test_acc = 0.0
        patience_counter = 0
        max_patience = self.xor_config.patience if self.xor_config.early_stopping else float('inf')
        
        for epoch in tqdm(range(self.config.epochs), desc="Training"):
            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            
            # Gradient clipping to prevent explosion
            if self.xor_config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.xor_config.gradient_clipping
                )
            
            loss.backward()
            self.optimizer.step()
            
            self.train_losses.append(loss.item())
            
            # Periodic evaluation
            if (epoch + 1) % 50 == 0:
                test_loss, test_acc = self._evaluate(X_test, y_test)
                self.test_losses.append(test_loss)
                self.test_accuracies.append(test_acc)
                
                print(f'Epoch [{epoch+1:3d}/{self.config.epochs}] | '
                      f'Train Loss: {loss.item():.4f} | '
                      f'Test Loss: {test_loss:.4f} | '
                      f'Test Acc: {test_acc:.4f}')
                
                # Update learning rate
                if self.scheduler:
                    if self.config.scheduler == 'plateau':
                        self.scheduler.step(test_loss)
                    else:
                        self.scheduler.step()
                
                # Early stopping check
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    patience_counter = 0
                else:
                    patience_counter += 50  # Since we check every 50 epochs
                
                if self.xor_config.early_stopping and patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Final evaluation
        print("-" * 60)
        final_test_loss, final_test_acc = self._evaluate(X_test, y_test)
        
        print(f'Final Test Accuracy: {final_test_acc:.4f}')
        
        # Compute per-class accuracy
        class_accuracies = self._compute_class_accuracies(X_test, y_test)
        for class_label, acc in class_accuracies.items():
            print(f'Class {class_label} Accuracy: {acc:.4f}')
        
        print("=" * 60)
        
        # Save visualizations
        if self.xor_config.save_plots:
            self._save_visualizations(X_test, y_test)
        
        # Return results
        results = {
            'final_accuracy': final_test_acc,
            'best_accuracy': best_test_acc,
            'final_loss': final_test_loss,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies,
            'class_accuracies': class_accuracies,
            'config': self.config.__dict__,
            'learned_prototypes': self._get_learned_parameters()
        }
        
        return results
    
    def _evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> Tuple[float, float]:
        """Evaluate model on test data"""
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test)
            test_loss = self.criterion(test_outputs, y_test)
            test_pred = (torch.sigmoid(test_outputs) > 0.5).float()
            test_acc = (test_pred == y_test).float().mean()
        self.model.train()
        return test_loss.item(), test_acc.item()
    
    def _compute_class_accuracies(self, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[int, float]:
        """Compute per-class accuracies"""
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test)
            test_pred = (torch.sigmoid(test_outputs) > 0.5).float()
            
            class_accs = {}
            for class_label in [0, 1]:
                class_mask = y_test.squeeze() == class_label
                if class_mask.sum() > 0:
                    class_acc = (test_pred[class_mask] == y_test[class_mask]).float().mean()
                    class_accs[class_label] = class_acc.item()
                else:
                    class_accs[class_label] = 0.0
        
        self.model.train()
        return class_accs
    
    def _save_visualizations(self, X_test: torch.Tensor, y_test: torch.Tensor):
        """Save visualization plots"""
        print("Generating visualizations...")
        
        # For visualization, we need the data on the same device as the model
        # The plot_decision_boundary function handles device management internally
        plot_decision_boundary(
            self.model, X_test, y_test,
            title="Tversky XOR Decision Boundary",
            save_path="xor_decision_boundary.png"
        )
        
        # Prototype and feature bank visualization
        prototypes = self.model.tversky.get_prototypes()
        feature_bank = self.model.tversky.get_feature_bank()
        
        # Handle feature bank type
        if isinstance(feature_bank, list):
            feature_bank_for_plot = feature_bank[0]
        else:
            feature_bank_for_plot = feature_bank
        
        plot_prototypes(
            prototypes, feature_bank_for_plot,
            title="Learned Prototypes and Feature Bank",
            save_path="xor_prototypes.png"
        )
        
        # Training curves
        extended_test_losses = []
        extended_test_accs = []
        for i in range(len(self.train_losses)):
            if (i + 1) % 50 == 0:
                idx = ((i + 1) // 50) - 1
                if idx < len(self.test_losses):
                    extended_test_losses.append(self.test_losses[idx])
                    extended_test_accs.append(self.test_accuracies[idx])
            else:
                if extended_test_losses:
                    extended_test_losses.append(extended_test_losses[-1])
                    extended_test_accs.append(extended_test_accs[-1])
        
        plot_training_curves(
            self.train_losses, extended_test_losses,
            None, extended_test_accs,
            title="XOR Training Curves",
            save_path="xor_training_curves.png"
        )
        
        print("Plots saved successfully!")
    
    def _get_learned_parameters(self) -> Dict[str, Any]:
        """Get learned prototypes and feature bank"""
        prototypes = self.model.tversky.get_prototypes()
        feature_bank = self.model.tversky.get_feature_bank()
        
        return {
            'prototypes': prototypes.cpu().numpy().tolist(),
            'feature_bank': feature_bank.cpu().numpy().tolist() if not isinstance(feature_bank, list) 
                           else [fb.cpu().numpy().tolist() for fb in feature_bank]
        }
