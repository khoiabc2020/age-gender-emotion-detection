"""
Threshold Optimization Script
Tối ưu threshold cho các classification tasks (Gender, Emotion) khi dataset imbalanced
Dựa trên validation set để tìm threshold tốt nhất cho F1 score
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.network import MultiTaskModel
from src.data.dataset import MultiTaskDataset
from torch.utils.data import DataLoader


def get_predictions_and_probs(model, dataloader, device):
    """Get predictions and probabilities from model"""
    model.eval()
    all_probs = {'gender': [], 'emotion': []}
    all_targets = {'gender': [], 'emotion': []}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Getting predictions'):
            images = batch['image'].to(device)
            gender_target = batch['gender'].to(device)
            emotion_target = batch['emotion'].to(device)
            
            gender_logits, _, emotion_logits = model(images)
            
            # Get probabilities
            gender_probs = torch.softmax(gender_logits, dim=1)
            emotion_probs = torch.softmax(emotion_logits, dim=1)
            
            all_probs['gender'].extend(gender_probs.cpu().numpy())
            all_targets['gender'].extend(gender_target.cpu().numpy())
            all_probs['emotion'].extend(emotion_probs.cpu().numpy())
            all_targets['emotion'].extend(emotion_target.cpu().numpy())
    
    return {
        'gender': (np.array(all_probs['gender']), np.array(all_targets['gender'])),
        'emotion': (np.array(all_probs['emotion']), np.array(all_targets['emotion']))
    }


def optimize_threshold(probs, targets, task_name='gender'):
    """
    Tìm threshold tối ưu cho F1 score
    
    Args:
        probs: Probability predictions (N, num_classes)
        targets: True labels (N,)
        task_name: 'gender' or 'emotion'
    
    Returns:
        best_threshold: Threshold tốt nhất
        best_metrics: Metrics tại threshold tốt nhất
    """
    num_classes = probs.shape[1]
    best_f1 = 0
    best_threshold = None
    best_metrics = None
    
    # For binary classification (gender), optimize threshold for positive class
    if num_classes == 2:
        positive_probs = probs[:, 1]  # Probability of positive class
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        
        for threshold in thresholds:
            preds = (positive_probs >= threshold).astype(int)
            f1 = f1_score(targets, preds, average='binary')
            precision = precision_score(targets, preds, average='binary', zero_division=0)
            recall = recall_score(targets, preds, average='binary', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'f1': float(f1),
                    'precision': float(precision),
                    'recall': float(recall),
                    'threshold': float(threshold)
                }
    
    # For multi-class (emotion), optimize per-class thresholds
    else:
        best_thresholds = []
        best_metrics_per_class = []
        
        for class_idx in range(num_classes):
            class_probs = probs[:, class_idx]
            class_targets = (targets == class_idx).astype(int)
            
            best_f1_class = 0
            best_threshold_class = 0.5
            
            thresholds = np.arange(0.1, 0.9, 0.05)
            for threshold in thresholds:
                preds = (class_probs >= threshold).astype(int)
                if np.sum(preds) > 0:  # Avoid division by zero
                    f1 = f1_score(class_targets, preds, average='binary', zero_division=0)
                    if f1 > best_f1_class:
                        best_f1_class = f1
                        best_threshold_class = threshold
            
            best_thresholds.append(best_threshold_class)
            best_metrics_per_class.append({
                'class': class_idx,
                'threshold': float(best_threshold_class),
                'f1': float(best_f1_class)
            })
        
        best_threshold = best_thresholds
        best_metrics = {
            'per_class': best_metrics_per_class,
            'average_f1': float(np.mean([m['f1'] for m in best_metrics_per_class]))
        }
    
    return best_threshold, best_metrics


def main():
    parser = argparse.ArgumentParser(description='Optimize classification thresholds')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed data directory')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split to use (val or test)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output', type=str, default='optimal_thresholds.json',
                        help='Output JSON file for thresholds')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    print("=" * 60)
    print("Threshold Optimization")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = MultiTaskModel()
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = MultiTaskDataset(args.data_dir, split=args.split, use_augmentation=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get predictions
    print("\nGetting predictions...")
    results = get_predictions_and_probs(model, dataloader, device)
    
    # Optimize thresholds
    print("\nOptimizing thresholds...")
    
    # Gender threshold
    gender_probs, gender_targets = results['gender']
    gender_threshold, gender_metrics = optimize_threshold(
        gender_probs, gender_targets, 'gender'
    )
    
    print(f"\n📊 Gender Threshold Optimization:")
    print(f"  Best threshold: {gender_threshold:.3f}")
    print(f"  F1 Score: {gender_metrics['f1']:.4f}")
    print(f"  Precision: {gender_metrics['precision']:.4f}")
    print(f"  Recall: {gender_metrics['recall']:.4f}")
    
    # Emotion thresholds
    emotion_probs, emotion_targets = results['emotion']
    emotion_thresholds, emotion_metrics = optimize_threshold(
        emotion_probs, emotion_targets, 'emotion'
    )
    
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print(f"\n📊 Emotion Threshold Optimization:")
    print(f"  Average F1: {emotion_metrics['average_f1']:.4f}")
    print(f"  Per-class thresholds:")
    for i, (name, threshold) in enumerate(zip(emotion_names, emotion_thresholds)):
        print(f"    {name}: {threshold:.3f}")
    
    # Save results
    output_data = {
        'gender': {
            'threshold': float(gender_threshold),
            'metrics': gender_metrics
        },
        'emotion': {
            'thresholds': [float(t) for t in emotion_thresholds],
            'metrics': emotion_metrics
        }
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Optimal thresholds saved to: {output_path}")
    print("\n" + "=" * 60)
    print("Usage: Use these thresholds in inference for better results")
    print("=" * 60)


if __name__ == "__main__":
    main()

