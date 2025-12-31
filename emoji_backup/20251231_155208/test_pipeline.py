"""
Test Pipeline - Kiểm tra toàn bộ pipeline từ preprocessing đến inference
Giai đoạn 1: Validation scripts
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.data.dataset import MultiTaskDataset, get_dataloaders
from src.models.network import MultiTaskModel, MultiTaskLoss


def test_preprocessing():
    """Test 1: Kiểm tra preprocessing đã chạy chưa"""
    print("=" * 60)
    print("TEST 1: Preprocessing Check")
    print("=" * 60)
    
    data_dir = Path(__file__).parent.parent / "data" / "processed" / "utkface"
    
    if not data_dir.exists():
        print(f"❌ FAILED: Processed data not found at: {data_dir}")
        print("   Please run: python src/data/preprocess.py")
        return False
    
    # Check splits
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"❌ FAILED: {split} directory not found")
            return False
        
        # Count images
        images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        if len(images) == 0:
            # Check subdirectories (for FER2013 structure)
            subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
            total = sum(len(list(d.glob("*.jpg")) + list(d.glob("*.png"))) for d in subdirs)
            print(f"  {split}: {total} images")
        else:
            print(f"  {split}: {len(images)} images")
    
    print("✅ PASSED: Preprocessing data found")
    return True


def test_dataset():
    """Test 2: Kiểm tra Dataset và DataLoader"""
    print("\n" + "=" * 60)
    print("TEST 2: Dataset & DataLoader")
    print("=" * 60)
    
    data_dir = Path(__file__).parent.parent / "data" / "processed" / "utkface"
    
    if not data_dir.exists():
        print("❌ SKIPPED: Preprocessing data not found")
        return False
    
    try:
        # Test dataset
        dataset = MultiTaskDataset(data_dir, split='train')
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Test sample
        sample = dataset[0]
        print(f"✅ Sample loaded:")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Age: {sample['age'].item()}")
        print(f"   Gender: {sample['gender'].item()}")
        print(f"   Emotion: {sample['emotion'].item()}")
        
        # Test DataLoader
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir, batch_size=4, num_workers=0
        )
        print(f"✅ DataLoaders created:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test batch
        batch = next(iter(train_loader))
        print(f"✅ Batch loaded:")
        print(f"   Batch size: {batch['image'].shape[0]}")
        print(f"   Image shape: {batch['image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test 3: Kiểm tra Model Architecture"""
    print("\n" + "=" * 60)
    print("TEST 3: Model Architecture")
    print("=" * 60)
    
    try:
        model = MultiTaskModel()
        print(f"✅ Model created")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        gender_logits, age_pred, emotion_logits = model(x)
        
        print(f"✅ Forward pass successful:")
        print(f"   Gender logits: {gender_logits.shape}")
        print(f"   Age prediction: {age_pred.shape}")
        print(f"   Emotion logits: {emotion_logits.shape}")
        
        # Test loss
        criterion = MultiTaskLoss()
        targets = (
            torch.randint(0, 2, (2,)),  # gender
            torch.randn(2) * 20 + 30,   # age
            torch.randint(0, 7, (2,))   # emotion
        )
        predictions = (gender_logits, age_pred, emotion_logits)
        losses = criterion(predictions, targets)
        
        print(f"✅ Loss calculation successful:")
        for key, value in losses.items():
            print(f"   {key}: {value.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test 4: Kiểm tra Training Step"""
    print("\n" + "=" * 60)
    print("TEST 4: Training Step")
    print("=" * 60)
    
    data_dir = Path(__file__).parent.parent / "data" / "processed" / "utkface"
    
    if not data_dir.exists():
        print("❌ SKIPPED: Preprocessing data not found")
        return False
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create model
        model = MultiTaskModel().to(device)
        criterion = MultiTaskLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Create small dataloader
        train_loader, _, _ = get_dataloaders(
            data_dir, batch_size=4, num_workers=0
        )
        
        # One training step
        model.train()
        batch = next(iter(train_loader))
        
        images = batch['image'].to(device)
        gender_target = batch['gender'].to(device)
        age_target = batch['age'].to(device)
        emotion_target = batch['emotion'].to(device)
        
        # Forward
        optimizer.zero_grad()
        gender_logits, age_pred, emotion_logits = model(images)
        
        predictions = (gender_logits, age_pred, emotion_logits)
        targets = (gender_target, age_target, emotion_target)
        
        losses = criterion(predictions, targets)
        
        # Backward
        losses['total'].backward()
        optimizer.step()
        
        print(f"✅ Training step successful:")
        print(f"   Loss: {losses['total'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("GIAI ĐOẠN 1: TEST PIPELINE")
    print("=" * 60)
    print("\nKiểm tra toàn bộ pipeline từ preprocessing đến training...\n")
    
    results = []
    
    # Run tests
    results.append(("Preprocessing", test_preprocessing()))
    results.append(("Dataset", test_dataset()))
    results.append(("Model", test_model()))
    results.append(("Training Step", test_training_step()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for training!")
        print("\nNext steps:")
        print("1. Run full training: python train.py --data_dir data/processed/utkface --epochs 50")
        print("2. Monitor: tensorboard --logdir checkpoints/logs")
        print("3. Convert to ONNX: python scripts/convert_to_onnx.py")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before training.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

