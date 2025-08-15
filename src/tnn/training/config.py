"""
Configuration classes for experiments
"""
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

@dataclass
class TverskyConfig:
    """Configuration for Tversky layer parameters"""
    num_prototypes: int = 8
    alpha: float = 0.5
    beta: float = 0.5
    theta: float = 1e-7
    intersection_reduction: Literal["product", "mean"] = "product"
    difference_reduction: Literal["ignorematch", "subtractmatch"] = "subtractmatch"
    feature_bank_init: Literal["ones", "random", "xavier"] = "xavier"
    prototype_init: Literal["random", "uniform", "normal", "xavier"] = "xavier"
    temperature: float = 1.0

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    
    # Experiment identification
    experiment_name: str = "tversky_resnet_mnist"
    run_name: Optional[str] = None
    
    # Model configuration
    architecture: Literal['resnet18', 'resnet50', 'resnet101', 'resnet152'] = 'resnet18'
    pretrained: bool = True
    frozen: bool = False
    use_tversky: bool = True
    
    # Dataset configuration
    dataset: Literal['mnist', 'nabirds'] = 'mnist'
    data_dir: str = './data'
    batch_size: int = 64
    num_workers: int = 4
    
    # Training configuration
    epochs: int = 50
    learning_rate: float = 0.01  # Higher default learning rate for Tversky layers
    weight_decay: float = 1e-4
    scheduler: Literal['none', 'cosine', 'step'] = 'cosine'
    scheduler_params: Dict[str, Any] = None
    
    # Optimizer configuration
    optimizer: Literal['adam', 'sgd'] = 'adam'
    momentum: float = 0.9  # For SGD
    
    # Evaluation configuration
    eval_every: int = 5  # Evaluate every N epochs
    save_checkpoints: bool = True
    checkpoint_dir: str = './checkpoints'
    
    # Tversky configuration
    tversky: TverskyConfig = None
    
    # Hardware configuration
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    mixed_precision: bool = True
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.tversky is None:
            self.tversky = TverskyConfig()
            
        if self.scheduler_params is None:
            if self.scheduler == 'cosine':
                self.scheduler_params = {'T_max': self.epochs}
            elif self.scheduler == 'step':
                self.scheduler_params = {'step_size': self.epochs // 3, 'gamma': 0.1}
            else:
                self.scheduler_params = {}
                
        if self.run_name is None:
            self.run_name = self._generate_run_name()
    
    def _generate_run_name(self) -> str:
        """Generate a descriptive run name"""
        components = [
            self.architecture,
            self.dataset,
            "tversky" if self.use_tversky else "linear",
            "pretrained" if self.pretrained else "scratch",
            "frozen" if self.frozen else "unfrozen"
        ]
        return "_".join(components)
    
    def get_num_classes(self) -> int:
        """Get number of classes for the dataset"""
        if self.dataset == 'mnist':
            return 10
        elif self.dataset == 'nabirds':
            return 400  # Approximate, will be updated with actual dataset
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if key == 'tversky':
                config_dict[key] = value.__dict__ if value else None
            else:
                config_dict[key] = value
        return config_dict
