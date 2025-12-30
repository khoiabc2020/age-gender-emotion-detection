"""
Evaluation Script - Đánh giá chi tiết model trên test set
Giai đoạn 1: Tuần 3 - Evaluation
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.network import MultiTaskModel
from src.data.dataset import MultiTaskDataset


def evaluate_model(model_path, data_dir, batch_size=32, device='cuda'):
    """
    Đánh giá model chi tiết trên test set
    
    Returns:
        metrics: Dictionary chứa tất cả metrics
    """
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = MultiTaskModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    if 'val_metrics' in checkpoint:
        print(f"Training metrics: {checkpoint['val_metrics']}")
    
    # Load test dataset
    print(f"\nLoading test dataset from: {data_dir}")
    test_dataset = MultiTaskDataset(data_dir, split='test', use_augmentation=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluation
    print("\nEvaluating on test set...")
    
    all_predictions = {
        'gender': [],
        'age': [],
        'emotion': []
    }
    all_targets = {
        'gender': [],
        'age': [],
        'emotion': []
    }
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            gender_target = batch['gender'].to(device)
            age_target = batch['age'].to(device)
            emotion_target = batch['emotion'].to(device)
            
            # Forward pass
            gender_logits, age_pred, emotion_logits = model(images)
            
            # Apply optimal thresholds nếu có
            if args.thresholds and Path(args.thresholds).exists():
                import json
                with open(args.thresholds) as f:
                    thresholds_data = json.load(f)
                
                # Gender threshold
                gender_probs = torch.softmax(gender_logits, dim=1)
                gender_threshold = thresholds_data['gender']['threshold']
                gender_pred = (gender_probs[:, 1] >= gender_threshold).long()
                all_predictions['gender'].extend(gender_pred.cpu().numpy())
                
                # Emotion thresholds (per-class)
                emotion_probs = torch.softmax(emotion_logits, dim=1)
                emotion_thresholds = thresholds_data['emotion']['thresholds']
                emotion_pred = torch.zeros(emotion_logits.size(0), dtype=torch.long, device=device)
                for i, threshold in enumerate(emotion_thresholds):
                    mask = emotion_probs[:, i] >= threshold
                    emotion_pred[mask] = i
                all_predictions['emotion'].extend(emotion_pred.cpu().numpy())
            else:
                # Default: argmax
                all_predictions['gender'].extend(gender_logits.argmax(dim=1).cpu().numpy())
                all_predictions['emotion'].extend(emotion_logits.argmax(dim=1).cpu().numpy())
            
            all_predictions['age'].extend(age_pred.squeeze().cpu().numpy())
            
            all_targets['gender'].extend(gender_target.cpu().numpy())
            all_targets['age'].extend(age_target.cpu().numpy())
            all_targets['emotion'].extend(emotion_target.cpu().numpy())
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # Gender accuracy
    gender_correct = np.sum(np.array(all_predictions['gender']) == np.array(all_targets['gender']))
    gender_acc = gender_correct / len(all_targets['gender']) * 100
    
    # Emotion accuracy
    emotion_correct = np.sum(np.array(all_predictions['emotion']) == np.array(all_targets['emotion']))
    emotion_acc = emotion_correct / len(all_targets['emotion']) * 100
    
    # Age MAE
    age_errors = np.abs(np.array(all_predictions['age']) - np.array(all_targets['age']))
    age_mae = np.mean(age_errors)
    age_std = np.std(age_errors)
    
    # Age RMSE
    age_rmse = np.sqrt(np.mean(age_errors ** 2))
    
    # Per-class metrics
    gender_names = ['Male', 'Female']
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # Gender per-class accuracy
    gender_per_class = {}
    for i, name in enumerate(gender_names):
        mask = np.array(all_targets['gender']) == i
        if np.sum(mask) > 0:
            correct = np.sum((np.array(all_predictions['gender']) == i) & mask)
            gender_per_class[name] = correct / np.sum(mask) * 100
    
    # Emotion per-class accuracy
    emotion_per_class = {}
    for i, name in enumerate(emotion_names):
        mask = np.array(all_targets['emotion']) == i
        if np.sum(mask) > 0:
            correct = np.sum((np.array(all_predictions['emotion']) == i) & mask)
            emotion_per_class[name] = correct / np.sum(mask) * 100
    
    # Age by range
    age_ranges = {
        '0-10': (0, 10),
        '11-20': (11, 20),
        '21-30': (21, 30),
        '31-40': (31, 40),
        '41-50': (41, 50),
        '51-60': (51, 60),
        '60+': (61, 200)
    }
    
    age_by_range = {}
    for range_name, (min_age, max_age) in age_ranges.items():
        mask = (np.array(all_targets['age']) >= min_age) & (np.array(all_targets['age']) <= max_age)
        if np.sum(mask) > 0:
            errors = age_errors[mask]
            age_by_range[range_name] = {
                'mae': float(np.mean(errors)),
                'count': int(np.sum(mask))
            }
    
    # Compile metrics
    metrics = {
        'overall': {
            'gender_accuracy': float(gender_acc),
            'emotion_accuracy': float(emotion_acc),
            'age_mae': float(age_mae),
            'age_std': float(age_std),
            'age_rmse': float(age_rmse)
        },
        'gender_per_class': gender_per_class,
        'emotion_per_class': emotion_per_class,
        'age_by_range': age_by_range,
        'test_samples': len(test_dataset)
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Gender Accuracy: {gender_acc:.2f}%")
    print(f"  Emotion Accuracy: {emotion_acc:.2f}%")
    print(f"  Age MAE: {age_mae:.2f} years")
    print(f"  Age STD: {age_std:.2f} years")
    print(f"  Age RMSE: {age_rmse:.2f} years")
    
    print(f"\n👤 Gender per-class:")
    for name, acc in gender_per_class.items():
        print(f"  {name}: {acc:.2f}%")
    
    print(f"\n😊 Emotion per-class:")
    for name, acc in emotion_per_class.items():
        print(f"  {name}: {acc:.2f}%")
    
    print(f"\n📅 Age by range:")
    for range_name, stats in age_by_range.items():
        print(f"  {range_name}: MAE={stats['mae']:.2f} years (n={stats['count']})")
    
    # Check if metrics meet targets
    print(f"\n🎯 Target Check:")
    targets_met = {
        'gender': gender_acc >= 92.0,
        'emotion': emotion_acc >= 75.0,
        'age': age_mae <= 4.5
    }
    
    for task, met in targets_met.items():
        status = "✅" if met else "❌"
        print(f"  {task.capitalize()}: {status}")
    
    print("=" * 60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for metrics')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--thresholds', type=str, default=None,
                        help='Path to optimal thresholds JSON file')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Evaluate
    metrics = evaluate_model(
        args.model_path,
        args.data_dir,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Save metrics
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✅ Metrics saved to: {output_path}")


if __name__ == "__main__":
    main()

