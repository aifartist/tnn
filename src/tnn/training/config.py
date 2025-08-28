"""
Configuration classes for experiments
"""
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any

@dataclass
class TverskyConfig:
    """Configuration for Tversky layer parameters"""
    num_prototypes: int = 8
    alpha: float = 0.5
    beta: float = 0.5
    theta: float = 1e-7
    intersection_reduction: Literal["product", "mean", "min", "max", "gmean", "softmin"] = "product"
    difference_reduction: Literal["ignorematch", "subtractmatch"] = "subtractmatch"
    feature_bank_init: Literal["ones", "random", "xavier"] = "xavier"
    prototype_init: Literal["random", "uniform", "normal", "xavier"] = "xavier"
    temperature: float = 1.0

@dataclass
class XORConfig:
    """Configuration for XOR toy problem experiments"""
    n_samples: int = 1000
    test_samples: int = 200
    hidden_dim: int = 8
    noise_std: float = 0.1
    save_plots: bool = True
    plot_dir: str = './info'
    early_stopping: bool = True
    patience: int = 50
    gradient_clipping: float = 1.0

@dataclass
class UnifiedConfig:
    """Unified configuration for all TNN model types"""

    # Core experiment settings
    model_type: Literal['xor', 'resnet', 'gpt2'] = 'xor'
    experiment_name: str = "tnn_experiment"
    run_name: Optional[str] = None

    # Common training settings
    epochs: int = 50
    learning_rate: float = 0.01
    batch_size: int = 64
    optimizer: Literal['adam', 'sgd', 'adamw'] = 'adam'
    weight_decay: float = 1e-4
    scheduler: Literal['none', 'cosine', 'step', 'plateau'] = 'cosine'
    scheduler_params: Optional[Dict[str, Any]] = None
    momentum: float = 0.9  # For SGD

    # ResNet-specific settings (previously in ExperimentConfig)
    architecture: Literal['resnet18', 'resnet50', 'resnet101', 'resnet152'] = 'resnet18'
    pretrained: bool = True
    frozen: bool = False
    use_tversky: bool = True
    dataset: Literal['mnist', 'nabirds'] = 'mnist'
    data_dir: str = './data'
    num_workers: int = 4
    eval_every: int = 5  # Evaluate every N epochs
    save_checkpoints: bool = True
    compile_model: bool = False

    # Common Tversky settings
    tversky: Optional[TverskyConfig] = None

    # Hardware and logging
    device: str = 'auto'
    checkpoint_dir: str = './checkpoints'
    results_dir: str = './results'
    mixed_precision: bool = True
    seed: int = 42

    # Model-specific configurations (only XOR now, ResNet params are now direct)
    xor_config: Optional[XORConfig] = None

    def __post_init__(self):
        """Post-initialization setup"""
        if self.tversky is None:
            self.tversky = TverskyConfig()

        if self.run_name is None:
            self.run_name = self._generate_run_name()

        # Initialize XOR config if needed
        if self.model_type == 'xor' and self.xor_config is None:
            self.xor_config = XORConfig()

        # Set up scheduler parameters
        if self.scheduler_params is None:
            if self.scheduler == 'cosine':
                self.scheduler_params = {'T_max': self.epochs}
            elif self.scheduler == 'step':
                self.scheduler_params = {'step_size': self.epochs // 3, 'gamma': 0.1}
            else:
                self.scheduler_params = {}

    def _generate_run_name(self) -> str:
        """Generate a descriptive run name"""
        if self.model_type == 'xor':
            prototypes = self.tversky.num_prototypes if self.tversky else 4
            return f"xor_{prototypes}p_ep{self.epochs}"
        elif self.model_type == 'resnet':
            components = [
                self.architecture,
                self.dataset,
                "tversky" if self.use_tversky else "linear",
                "pretrained" if self.pretrained else "scratch",
                "frozen" if self.frozen else "unfrozen"
            ]
            return "_".join(components)
        else:
            return f"{self.model_type}_experiment"

    def get_num_classes(self) -> int:
        """Get number of classes for the dataset"""
        if self.dataset == 'mnist':
            return 10
        elif self.dataset == 'nabirds':
            return 1011  # NABirds has 1011 classes (400 species but 1011 visual categories)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if key == 'tversky':
                config_dict[key] = value.__dict__ if value else None
            elif key == 'xor_config':
                config_dict[key] = value.__dict__ if value else None
            else:
                config_dict[key] = value
        return config_dict

