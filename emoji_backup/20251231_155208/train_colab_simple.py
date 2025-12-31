"""
Simple Training Script for Google Colab
Lightweight version without complex dependencies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path
import argparse
import sys
import time
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from src.models.mobileone import MobileOneMultiTaskModel
except ImportError:
    print("Warning: Could not import MobileOneMultiTaskModel")
    print("Using simplified model for testing")
    
    class MobileOneMultiTaskModel(nn.Module):
        def __init__(self, num_emotions=6, dropout_rate=0.3, width_multiplier=1.0):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.fc = nn.Linear(256, 128)
            self.dropout = nn.Dropout(dropout_rate)
            
            # Task heads
            self.gender_head = nn.Linear(128, 2)
            self.age_head = nn.Linear(128, 1)
            self.emotion_head = nn.Linear(128, num_emotions)
            
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.dropout(x)
            
            gender = self.gender_head(x)
            age = self.age_head(x)
            emotion = self.emotion_head(x)
            
            return gender, age, emotion


def get_data_loaders(data_dir, batch_size=64):
    """Load FER2013 dataset"""
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    data_path = Path(data_dir)
    
    # Try different paths
    if (data_path / 'train').exists():
        train_path = data_path / 'train'
    elif (data_path / 'Training').exists():
        train_path = data_path / 'Training'
    else:
        print(f"Looking for data in: {data_path}")
        print("Contents:", list(data_path.iterdir()) if data_path.exists() else "Not found")
        raise FileNotFoundError(f"Cannot find train data in {data_path}")
    
    print(f"Loading training data from: {train_path}")
    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    
    # Try to find test/val data
    test_path = None
    if (data_path / 'test').exists():
        test_path = data_path / 'test'
    elif (data_path / 'Testing').exists():
        test_path = data_path / 'Testing'
    
    if test_path and test_path.exists():
        print(f"Loading test data from: {test_path}")
        test_dataset = datasets.ImageFolder(test_path, transform=transform)
    else:
        print("No test data found, using train for evaluation")
        test_dataset = train_dataset
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    
    return train_loader, test_loader, len(train_dataset.classes)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        gender_out, age_out, emotion_out = model(inputs)
        
        # For simplicity, only use emotion loss
        loss = criterion(emotion_out, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Stats
        running_loss += loss.item()
        _, predicted = torch.max(emotion_out, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, device):
    """Evaluate model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            gender_out, age_out, emotion_out = model(inputs)
            
            _, predicted = torch.max(emotion_out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Simple Training for Colab')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    train_loader, test_loader, num_classes = get_data_loaders(args.data_dir, args.batch_size)
    
    # Create model
    print(f"\nCreating model...")
    model = MobileOneMultiTaskModel(num_emotions=num_classes, dropout_rate=0.3)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      patience=5, factor=0.5)
    
    # Training
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"{'='*60}\n")
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_acc = evaluate(model, test_loader, device)
        
        # Scheduler step
        scheduler.step(test_acc)
        
        # Print stats
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'num_classes': num_classes
            }
            
            torch.save(checkpoint, save_path / 'best_model.pth')
            print(f"✅ Saved best model (acc: {best_acc:.2f}%)")
    
    # Training complete
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {args.save_dir}/best_model.pth")
    print(f"{'='*60}")
    
    # Export to ONNX
    try:
        print(f"\nExporting to ONNX...")
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        onnx_path = Path(args.save_dir) / 'model.onnx'
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['gender', 'age', 'emotion'],
            dynamic_axes={'input': {0: 'batch_size'}}
        )
        print(f"✅ ONNX model saved to: {onnx_path}")
    except Exception as e:
        print(f"⚠️ Could not export to ONNX: {e}")
    
    # Save results
    import json
    results = {
        'best_accuracy': best_acc,
        'epochs': args.epochs,
        'training_time_minutes': elapsed / 60,
        'num_classes': num_classes
    }
    
    with open(Path(args.save_dir) / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ All done!")


if __name__ == '__main__':
    main()
