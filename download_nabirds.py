#!/usr/bin/env python3
"""
NABirds Dataset Download Script

This script downloads the NABirds dataset using Deeplake and caches it locally
in a standardized directory structure for easy access.
"""

import os
import sys
import argparse
from pathlib import Path

def download_nabirds(dataset_dir: str = "./datasets", force: bool = False):
    """
    Download NABirds dataset to local directory
    
    Args:
        dataset_dir: Base directory to store datasets
        force: Whether to re-download if dataset already exists
    """
    
    print("🦅 NABirds Dataset Downloader")
    print("=" * 50)
    
    try:
        import deeplake
        print("✅ Deeplake import successful")
    except ImportError:
        print("❌ Deeplake not found. Please install with: pip install 'deeplake<4'")
        sys.exit(1)
    
    # Create dataset directory structure
    nabirds_dir = Path(dataset_dir) / "nabirds"
    train_dir = nabirds_dir / "train"
    val_dir = nabirds_dir / "val"
    
    # Check if dataset already exists
    if train_dir.exists() and val_dir.exists() and not force:
        print(f"✅ NABirds dataset already exists in {nabirds_dir}")
        print("   Use --force to re-download")
        print_dataset_info(nabirds_dir)
        return str(nabirds_dir)
    
    # Create directories
    nabirds_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created dataset directory: {nabirds_dir}")
    
    # Download train split
    print("\n📥 Downloading NABirds training set...")
    try:
        print("   Loading from hub://activeloop/nabirds-dataset-train")
        train_ds = deeplake.load('hub://activeloop/nabirds-dataset-train')
        print(f"   Copying to {train_dir}...")
        train_ds.copy(str(train_dir))
        print(f"✅ Training set downloaded: {len(train_ds)} samples")
    except Exception as e:
        print(f"❌ Error downloading training set: {e}")
        return None
    
    # Download validation split
    print("\n📥 Downloading NABirds validation set...")
    try:
        print("   Loading from hub://activeloop/nabirds-dataset-val")
        val_ds = deeplake.load('hub://activeloop/nabirds-dataset-val')
        print(f"   Copying to {val_dir}...")
        val_ds.copy(str(val_dir))
        print(f"✅ Validation set downloaded: {len(val_ds)} samples")
    except Exception as e:
        print(f"❌ Error downloading validation set: {e}")
        return None
    
    create_dataset_info(nabirds_dir, train_ds, val_ds)
    
    print("\n🎉 NABirds dataset download complete!")
    print_dataset_info(nabirds_dir)
    
    return str(nabirds_dir)

def create_dataset_info(dataset_dir: Path, train_ds, val_ds):
    """Create a dataset info file with metadata"""
    info = {
        "dataset": "NABirds",
        "description": "North American Birds Dataset - 48,000 images of 400 bird species",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "num_classes": 1011,
        "splits": {
            "train": "train/",
            "val": "val/"
        },
        "citation": """
@inproceedings{nabirds2015,
  title={Building a Bird Recognition App and Large Scale Dataset With Citizen Scientists: The Fine Print in Fine-Grained Dataset Collection},
  author={Horn, Grant Van and Branson, Steve and Farrell, Ryan and Haber, Scott and Barry, Jessie and Ipeirotis, Panos and Belongie, Serge},
  booktitle={CVPR},
  year={2015}
}"""
    }
    
    import json
    info_file = dataset_dir / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"📄 Dataset info saved to {info_file}")

def print_dataset_info(dataset_dir: Path):
    """Print information about the downloaded dataset"""
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    
    print(f"\n📊 Dataset Information:")
    print(f"   Location: {dataset_dir}")
    print(f"   Training set: {train_dir}")
    print(f"   Validation set: {val_dir}")
    
    # Check sizes
    if train_dir.exists():
        try:
            import deeplake
            train_ds = deeplake.load(str(train_dir))
            print(f"   Train samples: {len(train_ds)}")
        except:
            print(f"   Train samples: Available")
    
    if val_dir.exists():
        try:
            import deeplake
            val_ds = deeplake.load(str(val_dir))
            print(f"   Val samples: {len(val_ds)}")
        except:
            print(f"   Val samples: Available")
    
    print(f"\n🚀 Ready to use with:")
    print(f"   python pipeline.py --model-type resnet --dataset nabirds --data-dir {dataset_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Download NABirds dataset for TNN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default ./datasets directory
  python download_nabirds.py
  
  # Download to custom directory
  python download_nabirds.py --dataset-dir /path/to/datasets
  
  # Force re-download
  python download_nabirds.py --force
"""
    )
    
    parser.add_argument(
        '--dataset-dir', 
        default='./datasets',
        help='Directory to store the dataset (default: ./datasets)'
    )
    
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force re-download even if dataset exists'
    )
    
    args = parser.parse_args()
    
    # Download dataset
    result = download_nabirds(args.dataset_dir, args.force)
    
    if result:
        print(f"\n✅ Download successful! Dataset ready at: {result}")
        sys.exit(0)
    else:
        print(f"\n❌ Download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
