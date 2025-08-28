#!/usr/bin/env python3
"""
Unified Training Pipeline for Tversky Neural Networks
Supports XOR toy problems, ResNet image classification, and future GPT-2 experiments
"""

import torch
import argparse
import os
import json
import sys
from datetime import datetime
from typing import Dict, Any, List

# Import from installed tnn package
from tnn.training import (
    UnifiedConfig, XORConfig, TverskyConfig,
    ExperimentFactory, Presets
)

def parse_args():
    """Parse command line arguments for unified training"""
    parser = argparse.ArgumentParser(
        description='Unified Training Pipeline for Tversky Neural Networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick XOR test
  python pipeline.py --model-type xor --preset xor_quick

  # XOR with custom parameters
  python pipeline.py --model-type xor --epochs 500 --lr 0.01 --prototypes 4

  # ResNet MNIST quick test
  python pipeline.py --model-type resnet --preset resnet_mnist_quick

  # ResNet MNIST full training
  python pipeline.py --model-type resnet --dataset mnist --architecture resnet18 --epochs 50

  # Table 1 reproduction
  python pipeline.py --model-type resnet --preset table1 --dataset mnist --architecture resnet18

  # NABirds paper reproduction
  python pipeline.py --model-type resnet --preset resnet_nabirds_paper
"""
    )

    # Core configuration
    parser.add_argument('--model-type', required=True, choices=['xor', 'resnet', 'gpt2'],
                       help='Type of model to train')
    parser.add_argument('--preset', type=str,
                       help='Use a preset configuration (xor_quick, xor_paper, resnet_mnist_quick, etc.)')
    parser.add_argument('--experiment-name', type=str,
                       help='Custom experiment name')

    # Common training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', default='cosine', choices=['none', 'cosine', 'step', 'plateau'],
                       help='Learning rate scheduler')

    # Scheduler-specific parameters
    parser.add_argument('--cosine-t-max', type=int,
                       help='T_max for cosine scheduler (defaults to epochs)')
    parser.add_argument('--step-size', type=int,
                       help='Step size for step scheduler (defaults to epochs//3)')
    parser.add_argument('--step-gamma', type=float, default=0.1,
                       help='Gamma for step scheduler')
    parser.add_argument('--plateau-patience', type=int, default=10,
                       help='Patience for plateau scheduler')
    parser.add_argument('--plateau-factor', type=float, default=0.1,
                       help='Factor for plateau scheduler')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # Tversky layer parameters
    parser.add_argument('--prototypes', type=int, default=8,
                       help='Number of prototypes in Tversky layer')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Tversky alpha parameter')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Tversky beta parameter')
    parser.add_argument('--intersection-reduction-function', default='product', choices=['product', 'mean', 'min', 'max', 'gmean', 'softmin'],
                       help='Intersection reduction function')

    # XOR-specific parameters
    parser.add_argument('--xor-samples', type=int, default=1000,
                       help='Number of XOR training samples')
    parser.add_argument('--xor-hidden-dim', type=int, default=8,
                       help='Hidden dimension for XOR network')
    parser.add_argument('--xor-noise', type=float, default=0.1,
                       help='Noise level for XOR data')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip saving plots for XOR')

    # ResNet-specific parameters
    parser.add_argument('--architecture', default='resnet18', choices=['resnet18', 'resnet50', 'resnet101', 'resnet152'],
                       help='ResNet architecture')
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'nabirds'],
                       help='Dataset for ResNet training')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                       help='Train from scratch')
    parser.add_argument('--frozen', action='store_true', default=False,
                       help='Freeze backbone parameters')
    parser.add_argument('--use-tversky', action='store_true', default=True,
                       help='Use Tversky projection layer')
    parser.add_argument('--use-linear', dest='use_tversky', action='store_false',
                       help='Use linear layer (baseline)')

    # Directories and output
    parser.add_argument('--data-dir', default='./data',
                       help='Data directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--results-dir', default='./results',
                       help='Results directory')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')

    # Special modes
    parser.add_argument('--table1', action='store_true',
                       help='Run all Table 1 configurations (ResNet only)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print configuration without training')

    return parser.parse_args()

def create_config_from_args(args) -> UnifiedConfig:
    """Create unified configuration from command line arguments"""

    # Check for preset first
    if args.preset:
        if args.preset == 'table1':
            # Special handling for table1 - will be handled separately
            pass
        else:
            try:
                config = Presets.get_preset(args.preset)
                # Override with any command line arguments
                if args.experiment_name:
                    config.experiment_name = args.experiment_name
                if args.epochs != 50:  # Only override if explicitly set
                    config.epochs = args.epochs
                return config
            except ValueError as e:
                print(f"Error loading preset: {e}")
                sys.exit(1)

    # Create Tversky config
    tversky_config = TverskyConfig(
        num_prototypes=args.prototypes,
        alpha=args.alpha,
        beta=args.beta,
        intersection_reduction=args.intersection_reduction_function
    )

    # Create scheduler parameters based on scheduler type and CLI args
    scheduler_params = None
    if args.scheduler == 'cosine':
        scheduler_params = {'T_max': args.cosine_t_max or args.epochs}
    elif args.scheduler == 'step':
        scheduler_params = {
            'step_size': args.step_size or (args.epochs // 3),
            'gamma': args.step_gamma
        }
    elif args.scheduler == 'plateau':
        scheduler_params = {
            'patience': args.plateau_patience,
            'factor': args.plateau_factor
        }

    # Create model-specific configurations
    if args.model_type == 'xor':
        xor_config = XORConfig(
            n_samples=args.xor_samples,
            hidden_dim=args.xor_hidden_dim,
            noise_std=args.xor_noise,
            save_plots=not args.no_plots
        )

        config = UnifiedConfig(
            model_type='xor',
            experiment_name=args.experiment_name or 'xor_experiment',
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            scheduler_params=scheduler_params,
            seed=args.seed,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            results_dir=args.results_dir,
            tversky=tversky_config,
            xor_config=xor_config
        )

    elif args.model_type == 'resnet':
        config = UnifiedConfig(
            model_type='resnet',
            experiment_name=args.experiment_name or 'resnet_experiment',
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            scheduler_params=scheduler_params,
            seed=args.seed,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            results_dir=args.results_dir,
            tversky=tversky_config,
            # ResNet-specific parameters (now direct fields)
            architecture=args.architecture,
            dataset=args.dataset,
            pretrained=args.pretrained,
            frozen=args.frozen,
            use_tversky=args.use_tversky,
            data_dir=args.data_dir,
            num_workers=4
        )

    else:
        raise ValueError(f"Model type {args.model_type} not supported yet")

    return config

def run_single_experiment(config: UnifiedConfig) -> Dict[str, Any]:
    """Run a single experiment with the given configuration"""

    print("=" * 80)
    print(f"TVERSKY NEURAL NETWORKS - {config.model_type.upper()} EXPERIMENT")
    print("=" * 80)
    print(f"Experiment: {config.experiment_name}")
    print(f"Run name: {config.run_name}")
    print(f"Model type: {config.model_type}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Device: {config.device}")

    if config.tversky:
        print(f"Tversky prototypes: {config.tversky.num_prototypes}")
        print(f"Tversky alpha: {config.tversky.alpha}, beta: {config.tversky.beta}")
        print(f"Tversky intersection reduction function: {config.tversky.intersection_reduction}")

    if config.model_type == 'resnet':
        print(f"Architecture: {config.architecture}")
        print(f"Dataset: {config.dataset}")
        print(f"Pretrained: {config.pretrained}")
        print(f"Frozen: {config.frozen}")
        print(f"Use Tversky: {config.use_tversky}")

    print("=" * 80)

    # Set random seed
    torch.manual_seed(config.seed)

    # Create experiment
    trainer, model, data_loaders = ExperimentFactory.create_experiment(config)

    # Run training
    results = trainer.train()

    return results

def run_table1_experiments(args) -> Dict[str, Any]:
    """Run all Table 1 configuration combinations"""

    print("=" * 80)
    print("RUNNING TABLE 1 REPRODUCTION EXPERIMENTS")
    print("=" * 80)
    print("This will run all combinations of pretrained/frozen with Tversky vs Linear")
    print(f"Architecture: {args.architecture}")
    print(f"Dataset: {args.dataset}")
    print("=" * 80)

    # Get all Table 1 configurations
    configs = Presets.table1_mnist(args.architecture)

    # Override dataset if specified
    if args.dataset != 'mnist':
        for config in configs:
            config.dataset = args.dataset

    all_results = {}

    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Running configuration {i+1}/{len(configs)}: {config.experiment_name}")
        print(f"{'='*60}")

        try:
            results = run_single_experiment(config)
            all_results[config.experiment_name] = results

            # Print summary
            if config.model_type == 'resnet':
                final_results = results.get('final_results', {})
                val_acc = final_results.get('validation', {}).get('accuracy', 'N/A')
                test_acc = final_results.get('test', {}).get('accuracy', 'N/A')
                best_val = results.get('best_val_accuracy', 'N/A')

                print(f"\nResults for {config.experiment_name}:")
                print(f"  Validation Accuracy: {val_acc}")
                print(f"  Test Accuracy: {test_acc}")
                print(f"  Best Val Accuracy: {best_val}")
            else:
                final_acc = results.get('final_accuracy', 'N/A')
                best_acc = results.get('best_accuracy', 'N/A')
                print(f"\nResults for {config.experiment_name}:")
                print(f"  Final Accuracy: {final_acc}")
                print(f"  Best Accuracy: {best_acc}")

        except Exception as e:
            print(f"Error in configuration {config.experiment_name}: {e}")
            all_results[config.experiment_name] = {'error': str(e)}

    return all_results

def save_results(results: Dict[str, Any], config: UnifiedConfig, args) -> str:
    """Save experiment results to file"""
    os.makedirs(args.results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.table1:
        filename = f"table1_results_{config.model_type}_{args.dataset}_{args.architecture}_{timestamp}.json"
    else:
        filename = f"unified_experiment_{config.model_type}_{config.run_name}_{timestamp}.json"

    filepath = os.path.join(args.results_dir, filename)

    # Prepare results with metadata
    results_with_metadata = {
        'timestamp': timestamp,
        'config': config.__dict__ if hasattr(config, '__dict__') else str(config),
        'args': vars(args),
        'results': results
    }

    # Handle non-serializable objects
    def default_serializer(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

    with open(filepath, 'w') as f:
        json.dump(results_with_metadata, f, indent=2, default=default_serializer)

    print(f"\nResults saved to: {filepath}")
    return filepath

def print_summary_table(results: Dict[str, Any], model_type: str):
    """Print a summary table of results"""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)

    if model_type == 'xor':
        print(f"{'Configuration':<30} {'Final Acc':<12} {'Best Acc':<12} {'Final Loss':<12}")
        print("-"*80)

        for config_name, result in results.items():
            if 'error' in result:
                print(f"{config_name:<30} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
            else:
                final_acc = result.get('final_accuracy', 0)
                best_acc = result.get('best_accuracy', 0)
                final_loss = result.get('final_loss', 0)
                print(f"{config_name:<30} {final_acc:<12.4f} {best_acc:<12.4f} {final_loss:<12.4f}")

    elif model_type == 'resnet':
        print(f"{'Configuration':<30} {'Val Acc':<10} {'Test Acc':<10} {'Best Val':<10}")
        print("-"*80)

        for config_name, result in results.items():
            if 'error' in result:
                print(f"{config_name:<30} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
            else:
                final_results = result.get('final_results', {})
                val_acc = final_results.get('validation', {}).get('accuracy', 0)
                test_acc = final_results.get('test', {}).get('accuracy', 'N/A')
                best_val = result.get('best_val_accuracy', 0)

                print(f"{config_name:<30} {val_acc:<10.4f} {str(test_acc):<10} {best_val:<10.4f}")

def main():
    """Main function"""
    args = parse_args()

    print("🚀 Tversky Neural Networks - Unified Training Pipeline")
    print("=" * 80)

    # Handle dry run
    if args.dry_run:
        if args.table1:
            configs = Presets.table1_mnist(args.architecture)
            print("Table 1 configurations that would be run:")
            for config in configs:
                print(f"  - {config.experiment_name}")
        else:
            config = create_config_from_args(args)
            print("Configuration that would be used:")
            print(json.dumps(config.__dict__, indent=2, default=str))
        return

    # Run experiments
    if args.table1:
        # Create a dummy config just for metadata
        dummy_config = create_config_from_args(args)
        results = run_table1_experiments(args)
        print_summary_table(results, dummy_config.model_type)
        save_results(results, dummy_config, args)
    else:
        config = create_config_from_args(args)
        results = run_single_experiment(config)

        # Print final summary
        if config.model_type == 'xor':
            final_acc = results.get('final_accuracy', 0)
            print(f"\n🎉 Training completed! Final accuracy: {final_acc:.4f}")
        elif config.model_type == 'resnet':
            final_results = results.get('final_results', {})
            val_acc = final_results.get('validation', {}).get('accuracy', 0)
            print(f"\n🎉 Training completed! Final validation accuracy: {val_acc:.4f}")

        save_results(results, config, args)

    print("\n✅ Pipeline execution completed!")

if __name__ == "__main__":
    main()
