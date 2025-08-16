"""
Preset configurations for common TNN experiments
"""
from .config import UnifiedConfig, XORConfig, ExperimentConfig, TverskyConfig

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
        resnet_config = ExperimentConfig(
            architecture='resnet18',
            dataset='mnist',
            pretrained=True,
            frozen=False,
            use_tversky=True,
            epochs=2,  # Quick test
            batch_size=64
        )
        
        return UnifiedConfig(
            model_type='resnet',
            experiment_name='resnet_mnist_quick',
            epochs=2,
            learning_rate=0.01,
            batch_size=64,
            tversky=TverskyConfig(num_prototypes=8, alpha=0.5, beta=0.5),
            resnet_config=resnet_config
        )
    
    @staticmethod
    def resnet_mnist_paper() -> UnifiedConfig:
        """ResNet MNIST configuration for paper reproduction"""
        resnet_config = ExperimentConfig(
            architecture='resnet18',
            dataset='mnist',
            pretrained=True,
            frozen=False,
            use_tversky=True,
            epochs=50,
            batch_size=64
        )
        
        return UnifiedConfig(
            model_type='resnet',
            experiment_name='resnet_mnist_paper',
            epochs=50,
            learning_rate=0.01,
            batch_size=64,
            tversky=TverskyConfig(num_prototypes=8, alpha=0.5, beta=0.5),
            resnet_config=resnet_config
        )
    
    @staticmethod
    def resnet_nabirds_paper() -> UnifiedConfig:
        """ResNet NABirds configuration for paper reproduction"""
        resnet_config = ExperimentConfig(
            architecture='resnet50',  # Paper uses ResNet-50 for NABirds
            dataset='nabirds',
            pretrained=True,
            frozen=True,  # Paper typically uses frozen backbone for NABirds
            use_tversky=True,
            epochs=100,
            batch_size=32  # Smaller batch size for larger model
        )
        
        return UnifiedConfig(
            model_type='resnet',
            experiment_name='resnet_nabirds_paper',
            epochs=100,
            learning_rate=0.01,
            batch_size=32,
            tversky=TverskyConfig(num_prototypes=16, alpha=0.5, beta=0.5),  # More prototypes for complex dataset
            resnet_config=resnet_config
        )
    
    @staticmethod
    def table1_mnist(architecture: str = 'resnet18') -> list[UnifiedConfig]:
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
            resnet_config = ExperimentConfig(
                architecture=architecture,
                dataset='mnist',
                pretrained=pretrained,
                frozen=frozen,
                use_tversky=use_tversky,
                epochs=50,
                batch_size=64
            )
            
            config = UnifiedConfig(
                model_type='resnet',
                experiment_name=f'table1_mnist_{desc}',
                epochs=50,
                learning_rate=0.01,
                batch_size=64,
                tversky=base_tversky,
                resnet_config=resnet_config
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
