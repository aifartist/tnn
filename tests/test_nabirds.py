#!/usr/bin/env python3
"""
Test script for NABirds dataset integration with Deeplake
"""

import sys
import os
sys.path.append('src')

def test_nabirds_dataset():
    """Test NABirds dataset loading"""
    print("🧪 Testing NABirds Dataset Integration with Deeplake")
    print("=" * 60)
    
    try:
        # Test basic import
        from tnn.datasets import get_nabirds_loaders
        print("✅ Successfully imported NABirds dataset loader")
        
        # Test dataset loading with dummy data (fallback)
        print("\n📦 Testing dataset loading...")
        train_loader, val_loader, test_loader = get_nabirds_loaders(
            batch_size=4,  # Small batch for testing
            num_workers=0,  # No multiprocessing for testing
            use_local=False  # Stream from hub
        )
        print("✅ Successfully created data loaders")
        
        # Test data loading
        print("\n🔄 Testing data loading...")
        for i, (images, labels) in enumerate(train_loader):
            print(f"  Batch {i+1}:")
            print(f"    Images shape: {images.shape}")
            print(f"    Labels shape: {labels.shape}")
            print(f"    Image dtype: {images.dtype}")
            print(f"    Labels dtype: {labels.dtype}")
            print(f"    Label range: {labels.min()} - {labels.max()}")
            
            if i >= 2:  # Test first 3 batches
                break
        
        print("✅ Data loading test successful!")
        
        print("\n📊 Dataset Statistics:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test local caching option
        print("\n💾 Testing local caching option...")
        cached_train_loader, _, _ = get_nabirds_loaders(
            batch_size=4,
            num_workers=0,
            use_local=True,
            data_dir='./temp_cache'
        )
        print("✅ Local caching option works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_nabirds_dataset()
    if success:
        print("\n🎉 NABirds dataset integration test completed successfully!")
    else:
        print("\n💥 NABirds dataset integration test failed!")
        sys.exit(1)
