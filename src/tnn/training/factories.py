"""
Factory classes for unified model, data, and trainer creation
"""
import torch
from typing import Any, Tuple, Union
from torch.utils.data import DataLoader

from .config import UnifiedConfig
from .xor_trainer import XORTrainer, TverskyXORNet
from .trainer import ResNetTrainer
from ..models import get_resnet_model
from ..datasets import get_mnist_loaders, get_nabirds_loaders
from ..utils import get_xor_data

class ModelFactory:
    """Factory for creating different model types"""
    
    @staticmethod
    def create_model(config: UnifiedConfig) -> torch.nn.Module:
        """Create model based on configuration"""
        if config.model_type == 'xor':
            return TverskyXORNet(
                hidden_dim=config.xor_config.hidden_dim,
                num_prototypes=config.tversky.num_prototypes,
                alpha=config.tversky.alpha,
                beta=config.tversky.beta
            )
        elif config.model_type == 'resnet':
            resnet_config = config.resnet_config
            return get_resnet_model(
                architecture=resnet_config.architecture,
                num_classes=resnet_config.get_num_classes(),
                pretrained=resnet_config.pretrained,
                frozen=resnet_config.frozen,
                use_tversky=resnet_config.use_tversky,
                **config.tversky.__dict__
            )
        else:
            raise ValueError(f"Model type {config.model_type} not supported yet")

class DataFactory:
    """Factory for creating different data loaders"""
    
    @staticmethod
    def create_data(config: UnifiedConfig) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[DataLoader, DataLoader, DataLoader]]:
        """Create data loaders/tensors based on configuration"""
        if config.model_type == 'xor':
            # Return raw tensors for XOR (handled by XORTrainer)
            X_train, y_train = get_xor_data(
                config.xor_config.n_samples, 
                noise_std=config.xor_config.noise_std
            )
            X_test, y_test = get_xor_data(
                config.xor_config.test_samples, 
                noise_std=config.xor_config.noise_std
            )
            return (X_train, y_train, X_test, y_test)
        
        elif config.model_type == 'resnet':
            resnet_config = config.resnet_config
            if resnet_config.dataset == 'mnist':
                return get_mnist_loaders(
                    data_dir=resnet_config.data_dir,
                    batch_size=config.batch_size,
                    frozen=resnet_config.frozen,
                    pretrained=resnet_config.pretrained
                )
            elif resnet_config.dataset == 'nabirds':
                return get_nabirds_loaders(
                    data_dir=resnet_config.data_dir,
                    batch_size=config.batch_size,
                    frozen=resnet_config.frozen,
                    pretrained=resnet_config.pretrained
                )
            else:
                raise ValueError(f"Dataset {resnet_config.dataset} not supported")
        else:
            raise ValueError(f"Model type {config.model_type} not supported yet")

class TrainerFactory:
    """Factory for creating different trainer types"""
    
    @staticmethod
    def create_trainer(config: UnifiedConfig, model: torch.nn.Module = None, data_loaders = None) -> Union[XORTrainer, ResNetTrainer]:
        """Create trainer based on configuration"""
        if config.model_type == 'xor':
            # XORTrainer handles its own model and data creation
            return XORTrainer(config)
        
        elif config.model_type == 'resnet':
            if model is None or data_loaders is None:
                raise ValueError("ResNet trainer requires model and data_loaders")
            
            train_loader, val_loader, test_loader = data_loaders
            return ResNetTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config.resnet_config
            )
        else:
            raise ValueError(f"Model type {config.model_type} not supported yet")

class ExperimentFactory:
    """High-level factory for creating complete experiments"""
    
    @staticmethod
    def create_experiment(config: UnifiedConfig):
        """Create a complete experiment setup"""
        if config.model_type == 'xor':
            # XOR experiments are self-contained
            trainer = TrainerFactory.create_trainer(config)
            return trainer, None, None
        
        elif config.model_type == 'resnet':
            # ResNet experiments need separate model and data creation
            model = ModelFactory.create_model(config)
            data_loaders = DataFactory.create_data(config)
            trainer = TrainerFactory.create_trainer(config, model, data_loaders)
            return trainer, model, data_loaders
        
        else:
            raise ValueError(f"Model type {config.model_type} not supported yet")
