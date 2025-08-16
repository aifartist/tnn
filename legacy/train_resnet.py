#!/usr/bin/env python3
"""
Main training script for TverskyResNet image classification experiments
Backward-compatible wrapper for pipeline.py
"""

import argparse
import subprocess
import sys

def main():
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
    
    args = parser.parse_args()
    
    # Build pipeline.py command
    cmd = [
        sys.executable, 'pipeline.py',
        '--model-type', 'resnet',
        '--architecture', args.architecture,
        '--dataset', args.dataset,
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--batch-size', str(args.batch_size),
        '--optimizer', args.optimizer,
        '--scheduler', args.scheduler,
        '--prototypes', str(args.num_prototypes),
        '--alpha', str(args.alpha),
        '--beta', str(args.beta),
        '--data-dir', args.data_dir,
        '--checkpoint-dir', args.checkpoint_dir,
        '--results-dir', args.results_dir,
        '--device', args.device,
    ]
    
    if args.experiment_name != 'tversky_resnet':
        cmd.extend(['--experiment-name', args.experiment_name])
    
    if not args.pretrained:
        cmd.append('--no-pretrained')
    
    if args.frozen:
        cmd.append('--frozen')
        
    if not args.use_tversky:
        cmd.append('--use-linear')
    
    if args.table1:
        cmd.append('--table1')
    
    print("🔄 Running unified pipeline for ResNet training...")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute pipeline
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
