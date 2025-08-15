"""
Training infrastructure for ResNet experiments
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import json
from typing import Dict, Optional, Tuple, Any, Union
from tqdm import tqdm

from .config import ExperimentConfig
from .metrics import ClassificationMetrics, MetricsTracker

class ResNetTrainer:
    """
    Trainer class for ResNet image classification experiments
    Handles both Tversky and baseline models with comprehensive logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ExperimentConfig,
        test_loader: Optional[DataLoader] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # Setup device
        self.device = self._setup_device()
        self.model = self.model.to(self.device)
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup metrics tracking
        num_classes = config.get_num_classes()
        self.train_metrics = ClassificationMetrics(num_classes)
        self.val_metrics = ClassificationMetrics(num_classes)
        self.metrics_tracker = MetricsTracker()
        
        # Setup mixed precision if available
        self.scaler = None
        if config.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {config.mixed_precision and self.scaler is not None}")
        print(f"  Optimizer: {config.optimizer}")
        print(f"  Scheduler: {config.scheduler}")
        
        # Print model info if available
        if hasattr(self.model, 'print_model_info'):
            self.model.print_model_info()
        
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if self.config.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config.device)
        return device
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with different learning rates for backbone and classifier"""
        if hasattr(self.model, 'frozen') and self.model.frozen:
            # Only train classifier parameters
            trainable_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    trainable_params.append(param)
            
            if self.config.optimizer == 'adam':
                optimizer = optim.Adam(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            elif self.config.optimizer == 'sgd':
                optimizer = optim.SGD(
                    trainable_params,
                    lr=self.config.learning_rate,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay
                )
        else:
            # Train all parameters with optimized learning rates for Tversky models
            backbone_params = []
            tversky_params = []
            classifier_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                if 'prototype' in name or 'feature_bank' in name:
                    # Tversky layer parameters need higher learning rate
                    tversky_params.append(param)
                elif 'prototype_to_class' in name or 'fc' in name:
                    # Final classification layer
                    classifier_params.append(param)
                else:
                    # Backbone parameters
                    backbone_params.append(param)
            
            # Create parameter groups with different learning rates
            param_groups = []
            if backbone_params:
                param_groups.append({
                    'params': backbone_params, 
                    'lr': self.config.learning_rate * 0.1,  # Lower LR for pretrained backbone
                    'name': 'backbone'
                })
            if tversky_params:
                param_groups.append({
                    'params': tversky_params, 
                    'lr': self.config.learning_rate * 5,  # Higher LR for Tversky layer
                    'name': 'tversky'
                })
            if classifier_params:
                param_groups.append({
                    'params': classifier_params, 
                    'lr': self.config.learning_rate * 2,  # Medium LR for classifier
                    'name': 'classifier'
                })
            
            if not param_groups:
                # Fallback to all parameters
                param_groups = list(self.model.parameters())
            
            if self.config.optimizer == 'adam':
                optimizer = optim.Adam(
                    param_groups,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            elif self.config.optimizer == 'sgd':
                optimizer = optim.SGD(
                    param_groups,
                    lr=self.config.learning_rate,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    nesterov=True
                )
            else:
                raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
        return optimizer
        
    def _setup_scheduler(self) -> Optional[Union[optim.lr_scheduler.CosineAnnealingLR, optim.lr_scheduler.StepLR]]:
        """Setup learning rate scheduler"""
        if self.config.scheduler == 'none':
            return None
        elif self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **self.config.scheduler_params
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, **self.config.scheduler_params
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
            
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping for stability
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(output.detach(), target, loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return self.train_metrics.compute()
        
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.val_metrics.update(output, target, loss.item())
                
        return self.val_metrics.compute()
        
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        print(f"Starting training for {self.config.epochs} epochs...")
        print("=" * 60)
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = None
            if (epoch + 1) % self.config.eval_every == 0 or epoch == self.config.epochs - 1:
                val_metrics = self.validate()
                
                # Check for best model
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    if self.config.save_checkpoints:
                        self.save_checkpoint(epoch, is_best=True)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
                
            # Track metrics
            self.metrics_tracker.add_epoch(epoch, train_metrics, val_metrics)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch+1:3d}/{self.config.epochs}] ({epoch_time:.1f}s)")
            print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}")
            if val_metrics:
                print(f"  Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}")
            print()
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s")
        
        # Final evaluation
        final_results = self.evaluate_final()
        
        return {
            'config': self.config.to_dict(),
            'metrics': self.metrics_tracker.to_dict(),
            'best_val_accuracy': best_val_acc,
            'final_results': final_results,
            'training_time': total_time
        }
        
    def evaluate_final(self) -> Dict[str, Any]:
        """Final evaluation on validation and test sets"""
        results = {}
        
        # Validation results
        val_metrics = self.validate()
        results['validation'] = val_metrics
        
        # Test results if available
        if self.test_loader:
            test_metrics = ClassificationMetrics(self.config.get_num_classes())
            self.model.eval()
            
            with torch.no_grad():
                for data, target in tqdm(self.test_loader, desc="Testing"):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            output = self.model(data)
                            loss = self.criterion(output, target)
                    else:
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    
                    test_metrics.update(output, target, loss.item())
            
            results['test'] = test_metrics.compute()
        
        return results
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'metrics': self.metrics_tracker.to_dict()
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"{self.config.run_name}_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                f"{self.config.run_name}_best.pt"
            )
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['epoch']
