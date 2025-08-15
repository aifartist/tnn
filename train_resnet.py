#!/usr/bin/env python3
"""
Main training script for TverskyResNet image classification experiments
Reproduces Table 1 from the paper with MNIST and NABirds datasets
"""

import torch
import argparse
import os
import json
from datetime import datetime

# Import from installed tnn package
from tnn.models import get_resnet_model
from tnn.datasets import get_mnist_loaders, get_nabirds_loaders
from tnn.training import ResNetTrainer, ExperimentConfig, TverskyConfig

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train TverskyResNet for image classification')
    
    # Model configuration
    parser.add_argument('--architecture', default='resnet18', choices=['resnet18', 'resnet50', 'resnet101', 'resnet152'],
                       help='ResNet architecture')
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
    
    # Dataset configuration
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'nabirds'],
                       help='Dataset to use')
    parser.add_argument('--data-dir', default='./data',
                       help='Data directory')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', default='cosine', choices=['none', 'cosine', 'step'],
                       help='Learning rate scheduler')
    
    # Tversky configuration
    parser.add_argument('--num-prototypes', type=int, default=8,
                       help='Number of prototypes in Tversky layer')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Tversky alpha parameter')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Tversky beta parameter')
    
    # Experiment configuration
    parser.add_argument('--experiment-name', default='tversky_resnet',
                       help='Experiment name')
    parser.add_argument('--checkpoint-dir', default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--results-dir', default='./results',
                       help='Results directory')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    # Quick test modes
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (reduced epochs and data)')
    parser.add_argument('--table1', action='store_true',
                       help='Run all Table 1 configurations')
    
    return parser.parse_args()

def create_config(args) -> ExperimentConfig:
    """Create experiment configuration from arguments"""
    
    # Use the specified epochs directly (quick mode can be handled elsewhere if needed)
    epochs = args.epochs
    
    # Create Tversky config
    tversky_config = TverskyConfig(
        num_prototypes=args.num_prototypes,
        alpha=args.alpha,
        beta=args.beta
    )
    
    # Create main config
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        architecture=args.architecture,
        pretrained=args.pretrained,
        frozen=args.frozen,
        use_tversky=args.use_tversky,
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=epochs,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        tversky=tversky_config
    )
    
    return config

def run_single_experiment(config: ExperimentConfig) -> dict:
    """Run a single experiment with given configuration"""
    
    print("=" * 80)
    print(f"EXPERIMENT: {config.run_name}")
    print("=" * 80)
    print(f"Dataset: {config.dataset}")
    print(f"Architecture: {config.architecture}")
    print(f"Pretrained: {config.pretrained}")
    print(f"Frozen: {config.frozen}")
    print(f"Classifier: {'Tversky' if config.use_tversky else 'Linear'}")
    if config.use_tversky:
        print(f"Prototypes: {config.tversky.num_prototypes}")
        print(f"Alpha: {config.tversky.alpha}, Beta: {config.tversky.beta}")
    print("=" * 80)
    
    # Load dataset
    if config.dataset == 'mnist':
        train_loader, val_loader, test_loader = get_mnist_loaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            frozen=config.frozen,
            pretrained=config.pretrained
        )
    elif config.dataset == 'nabirds':
        train_loader, val_loader, test_loader = get_nabirds_loaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            frozen=config.frozen,
            pretrained=config.pretrained
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    # Create model
    model = get_resnet_model(
        architecture=config.architecture,
        num_classes=config.get_num_classes(),
        pretrained=config.pretrained,
        frozen=config.frozen,
        use_tversky=config.use_tversky,
        **config.tversky.__dict__
    )
    
    # Create trainer
    trainer = ResNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config
    )
    
    # Train model
    results = trainer.train()
    
    return results

def run_table1_experiments(base_config: ExperimentConfig, args):
    """Run all Table 1 configuration combinations"""
    
    print("Running Table 1 reproduction experiments...")
    print("This will test all combinations of pretrained/frozen with Tversky vs Linear")
    
    configurations = [
        # (pretrained, frozen, use_tversky, description)
        (True, True, True, "pretrained_frozen_tversky"),
        (True, True, False, "pretrained_frozen_linear"),
        (True, False, True, "pretrained_unfrozen_tversky"),
        (True, False, False, "pretrained_unfrozen_linear"),
        (False, False, True, "scratch_unfrozen_tversky"),
        (False, False, False, "scratch_unfrozen_linear"),
    ]
    
    all_results = {}
    
    for pretrained, frozen, use_tversky, desc in configurations:
        print(f"\n{'='*60}")
        print(f"Running configuration: {desc}")
        print(f"{'='*60}")
        
        # Create modified config
        config = ExperimentConfig(
            experiment_name=f"{base_config.experiment_name}_table1",
            run_name=f"{base_config.architecture}_{base_config.dataset}_{desc}",
            architecture=base_config.architecture,
            dataset=base_config.dataset,
            pretrained=pretrained,
            frozen=frozen,
            use_tversky=use_tversky,
            epochs=base_config.epochs,
            batch_size=base_config.batch_size,
            learning_rate=base_config.learning_rate,
            data_dir=base_config.data_dir,
            checkpoint_dir=base_config.checkpoint_dir,
            tversky=base_config.tversky
        )
        
        try:
            results = run_single_experiment(config)
            all_results[desc] = results
            
            # Print summary
            final_results = results['final_results']
            val_acc = final_results['validation']['accuracy']
            test_acc = final_results.get('test', {}).get('accuracy', 'N/A')
            
            print(f"\nResults for {desc}:")
            print(f"  Validation Accuracy: {val_acc:.4f}")
            print(f"  Test Accuracy: {test_acc}")
            print(f"  Best Val Accuracy: {results['best_val_accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error in configuration {desc}: {e}")
            all_results[desc] = {'error': str(e)}
    
    return all_results

def save_results(results: dict, args, config: ExperimentConfig):
    """Save experiment results"""
    os.makedirs(args.results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.table1:
        filename = f"table1_results_{config.dataset}_{config.architecture}_{timestamp}.json"
    else:
        filename = f"single_experiment_{config.run_name}_{timestamp}.json"
    
    filepath = os.path.join(args.results_dir, filename)
    
    # Add metadata
    results_with_metadata = {
        'timestamp': timestamp,
        'args': vars(args),
        'results': results
    }
    
    with open(filepath, 'w') as f:
        json.dump(results_with_metadata, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")
    return filepath

def main():
    """Main function"""
    args = parse_args()
    
    print("TverskyResNet Image Classification Training")
    print("==========================================")
    
    # Create base configuration
    config = create_config(args)
    
    # Run experiments
    if args.table1:
        results = run_table1_experiments(config, args)
        
        # Print comparison table
        print("\n" + "="*80)
        print("TABLE 1 RESULTS SUMMARY")
        print("="*80)
        print(f"{'Configuration':<30} {'Val Acc':<10} {'Test Acc':<10} {'Best Val':<10}")
        print("-"*80)
        
        for config_name, result in results.items():
            if 'error' in result:
                print(f"{config_name:<30} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
            else:
                val_acc = result['final_results']['validation']['accuracy']
                test_acc = result['final_results'].get('test', {}).get('accuracy', 'N/A')
                best_val = result['best_val_accuracy']
                
                print(f"{config_name:<30} {val_acc:<10.4f} {str(test_acc):<10} {best_val:<10.4f}")
        
    else:
        results = run_single_experiment(config)
    
    # Save results
    save_results(results, args, config)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
