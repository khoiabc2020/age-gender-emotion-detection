"""
Local Training Script - Train offline on your machine
Run: python train_local.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Import model architecture
import sys
sys.path.append(str(Path(__file__).parent))
from src.models.network import MultiTaskModel

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

DATA_DIR = Path(__file__).parent / 'data' / 'processed'
CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints' / 'local'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Data directory: {DATA_DIR}")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")

# ============================================================
# DATA LOADING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("\nLoading datasets...")
train_dataset = datasets.ImageFolder(DATA_DIR / 'train', transform=transform)
test_dataset = datasets.ImageFolder(DATA_DIR / 'test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ============================================================
# MODEL SETUP
# ============================================================

model = MultiTaskModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(loader), 100. * correct / total

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

best_acc = 0.0
training_history = []

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 60)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    
    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    
    print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Save history
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    })
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'test_loss': test_loss
        }
        torch.save(checkpoint, CHECKPOINT_DIR / 'best_model.pth')
        print(f"✓ Saved best model (acc: {best_acc:.2f}%)")

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    'timestamp': datetime.now().isoformat(),
    'device': DEVICE,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'best_accuracy': best_acc,
    'history': training_history
}

with open(CHECKPOINT_DIR / 'training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Best Test Accuracy: {best_acc:.2f}%")
print(f"Model saved to: {CHECKPOINT_DIR / 'best_model.pth'}")
print(f"Results saved to: {CHECKPOINT_DIR / 'training_results.json'}")
