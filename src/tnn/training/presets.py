"""
Preset configurations for common TNN experiments
"""
from typing import Literal
from tnn.training.config import UnifiedConfig, XORConfig, TverskyConfig

class Presets:
    """Common preset configurations for TNN experiments"""

    @staticmethod
    def xor_quick() -> UnifiedConfig:
        """Quick XOR test (fast convergence)"""
        return UnifiedConfig(
            model_type='xor',
            experiment_name='xor_quick_test',
            epochs=100,
            learning_rate=0.01,
            tversky=TverskyConfig(num_prototypes=4, alpha=0.5, beta=0.5),
            xor_config=XORConfig(n_samples=500, test_samples=100, hidden_dim=8)
        )

    @staticmethod
    def xor_paper() -> UnifiedConfig:
        """XOR configuration matching paper settings"""
        return UnifiedConfig(
            model_type='xor',
            experiment_name='xor_paper_reproduction',
            epochs=500,
            learning_rate=0.01,
            tversky=TverskyConfig(num_prototypes=4, alpha=0.5, beta=0.5),
            xor_config=XORConfig(n_samples=1000, test_samples=200, hidden_dim=8)
        )

    @staticmethod
    def resnet_mnist_quick() -> UnifiedConfig:
        """Quick ResNet MNIST test"""
        return UnifiedConfig(
            model_type='resnet',
            experiment_name='resnet_mnist_quick',
            epochs=2,
            learning_rate=0.01,
            batch_size=64,
            architecture='resnet18',
            dataset='mnist',
            pretrained=True,
            frozen=False,
            use_tversky=True,
            tversky=TverskyConfig(num_prototypes=8, alpha=0.5, beta=0.5)
        )

    @staticmethod
    def resnet_mnist_paper() -> UnifiedConfig:
        """ResNet MNIST configuration for paper reproduction"""
        return UnifiedConfig(
            model_type='resnet',
            experiment_name='resnet_mnist_paper',
            epochs=50,
            learning_rate=0.01,
            batch_size=64,
            architecture='resnet18',
            dataset='mnist',
            pretrained=True,
            frozen=False,
            use_tversky=True,
            tversky=TverskyConfig(num_prototypes=8, alpha=0.5, beta=0.5)
        )

    @staticmethod
    def resnet_nabirds_paper() -> UnifiedConfig:
        """ResNet NABirds configuration for paper reproduction"""
        return UnifiedConfig(
            model_type='resnet',
            experiment_name='resnet_nabirds_paper',
            epochs=100,
            learning_rate=0.001,  # Lower LR for NABirds
            batch_size=32,        # Smaller batch for NABirds
            architecture='resnet50',
            dataset='nabirds',
            pretrained=True,
            frozen=False,
            use_tversky=True,
            tversky=TverskyConfig(num_prototypes=16, alpha=0.5, beta=0.5)
        )

    @staticmethod
    def table1_mnist(architecture: Literal['resnet18', 'resnet50', 'resnet101', 'resnet152'] = 'resnet18') -> list[UnifiedConfig]:
        """Generate all Table 1 configurations for MNIST"""
        configs = []

        base_tversky = TverskyConfig(num_prototypes=8, alpha=0.5, beta=0.5)

        # All combinations from paper's Table 1
        combinations = [
            (True, True, True, "pretrained_frozen_tversky"),
            (True, True, False, "pretrained_frozen_linear"),
            (True, False, True, "pretrained_unfrozen_tversky"),
            (True, False, False, "pretrained_unfrozen_linear"),
            (False, False, True, "scratch_unfrozen_tversky"),
            (False, False, False, "scratch_unfrozen_linear"),
        ]

        for pretrained, frozen, use_tversky, desc in combinations:
            config = UnifiedConfig(
                model_type='resnet',
                experiment_name=f'table1_mnist_{desc}',
                epochs=50,
                learning_rate=0.01,
                batch_size=64,
                architecture=architecture,
                dataset='mnist',
                pretrained=pretrained,
                frozen=frozen,
                use_tversky=use_tversky,
                tversky=base_tversky
            )
            configs.append(config)

        return configs

    @staticmethod
    def get_preset(name: str) -> UnifiedConfig:
        """Get preset configuration by name"""
        presets = {
            'xor_quick': Presets.xor_quick,
            'xor_paper': Presets.xor_paper,
            'resnet_mnist_quick': Presets.resnet_mnist_quick,
            'resnet_mnist_paper': Presets.resnet_mnist_paper,
            'resnet_nabirds_paper': Presets.resnet_nabirds_paper,
        }

        if name not in presets:
            available = ', '.join(presets.keys())
            raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")

        return presets[name]()
