#!/usr/bin/env python3
"""
XOR toy problem to test Tversky Neural Networks
Backward-compatible wrapper for pipeline.py
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Train Tversky XOR Network')
    parser.add_argument('--samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=8, help='Hidden dimension')
    parser.add_argument('--prototypes', type=int, default=4, help='Number of prototypes')
    parser.add_argument('--alpha', type=float, default=0.5, help='Tversky alpha parameter')
    parser.add_argument('--beta', type=float, default=0.5, help='Tversky beta parameter')
    parser.add_argument('--no-plots', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    # Build pipeline.py command
    cmd = [
        sys.executable, 'pipeline.py',
        '--model-type', 'xor',
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--prototypes', str(args.prototypes),
        '--alpha', str(args.alpha),
        '--beta', str(args.beta),
        '--xor-samples', str(args.samples),
        '--xor-hidden-dim', str(args.hidden_dim),
    ]
    
    if args.no_plots:
        cmd.append('--no-plots')
    
    print("🔄 Running unified pipeline for XOR training...")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute pipeline
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
