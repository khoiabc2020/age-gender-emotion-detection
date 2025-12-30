"""
Script tổng hợp kết quả training
Tạo báo cáo chi tiết về quá trình training và metrics
"""
import torch
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_checkpoint(checkpoint_path):
    """Load checkpoint và trả về thông tin"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint

def parse_tensorboard_logs(log_dir):
    """Parse TensorBoard logs để lấy metrics"""
    try:
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
        
        metrics = {}
        for tag in ea.Tags()['scalars']:
            scalar_events = ea.Scalars(tag)
            values = [s.value for s in scalar_events]
            steps = [s.step for s in scalar_events]
            metrics[tag] = {'values': values, 'steps': steps}
        
        return metrics
    except Exception as e:
        print(f"Warning: Could not parse TensorBoard logs: {e}")
        return {}

def create_summary_report(checkpoints_dir, output_file='training_summary.json'):
    """Tạo báo cáo tổng hợp training"""
    checkpoints_dir = Path(checkpoints_dir)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'checkpoints_location': str(checkpoints_dir.absolute()),
        'best_model': {},
        'all_checkpoints': [],
        'training_logs': {}
    }
    
    # Load best model
    best_model_path = checkpoints_dir / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = load_checkpoint(best_model_path)
        summary['best_model'] = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_loss': float(checkpoint.get('val_loss', 0)),
            'val_metrics': checkpoint.get('val_metrics', {}),
            'file_size_mb': best_model_path.stat().st_size / (1024 * 1024)
        }
        print(f"\n[SUCCESS] Best Model Found!")
        print(f"  Epoch: {summary['best_model']['epoch']}")
        print(f"  Validation Loss: {summary['best_model']['val_loss']:.4f}")
        print(f"  File Size: {summary['best_model']['file_size_mb']:.2f} MB")
        if summary['best_model']['val_metrics']:
            print("  Metrics:")
            for key, value in summary['best_model']['val_metrics'].items():
                print(f"    {key}: {value:.2f}")
    
    # List all checkpoints
    checkpoint_files = list(checkpoints_dir.glob('checkpoint_epoch_*.pth'))
    for ckpt_file in sorted(checkpoint_files):
        try:
            checkpoint = load_checkpoint(ckpt_file)
            summary['all_checkpoints'].append({
                'file': ckpt_file.name,
                'epoch': checkpoint.get('epoch', 'N/A'),
                'val_loss': float(checkpoint.get('val_loss', 0)),
                'file_size_mb': ckpt_file.stat().st_size / (1024 * 1024)
            })
        except Exception as e:
            print(f"Warning: Could not load {ckpt_file}: {e}")
    
    # Parse TensorBoard logs
    logs_dir = checkpoints_dir / 'logs'
    if logs_dir.exists():
        summary['training_logs'] = parse_tensorboard_logs(logs_dir)
        print(f"\n[SUCCESS] Found TensorBoard logs with {len(summary['training_logs'])} metrics")
    
    # Save summary
    output_path = checkpoints_dir / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Summary saved to: {output_path}")
    
    return summary

def create_visualization(checkpoints_dir, output_file='training_plots.png'):
    """Tạo biểu đồ từ TensorBoard logs"""
    checkpoints_dir = Path(checkpoints_dir)
    logs_dir = checkpoints_dir / 'logs'
    
    if not logs_dir.exists():
        print("No TensorBoard logs found for visualization")
        return
    
    try:
        metrics = parse_tensorboard_logs(logs_dir)
        if not metrics:
            print("No metrics found in logs")
            return
        
        # Tạo subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Loss plots
        if 'Train/total_loss' in metrics:
            ax = axes[0, 0]
            data = metrics['Train/total_loss']
            ax.plot(data['steps'], data['values'], label='Train Loss')
            ax.set_title('Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        if 'Val/total_loss' in metrics:
            ax = axes[0, 1]
            data = metrics['Val/total_loss']
            ax.plot(data['steps'], data['values'], label='Val Loss', color='orange')
            ax.set_title('Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        # Accuracy plots
        if 'Val/gender_acc' in metrics:
            ax = axes[1, 0]
            data = metrics['Val/gender_acc']
            ax.plot(data['steps'], data['values'], label='Gender Accuracy', color='green')
            ax.set_title('Gender Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.legend()
            ax.grid(True)
        
        if 'Val/emotion_acc' in metrics:
            ax = axes[1, 1]
            data = metrics['Val/emotion_acc']
            ax.plot(data['steps'], data['values'], label='Emotion Accuracy', color='purple')
            ax.set_title('Emotion Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        output_path = checkpoints_dir / output_file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[SUCCESS] Visualization saved to: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Summarize training results')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='Path to checkpoints directory')
    parser.add_argument('--create_plots', action='store_true',
                        help='Create visualization plots')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Results Summary")
    print("=" * 60)
    
    # Create summary
    summary = create_summary_report(args.checkpoints_dir)
    
    # Create plots if requested
    if args.create_plots:
        print("\nCreating visualization plots...")
        create_visualization(args.checkpoints_dir)
    
    print("\n" + "=" * 60)
    print("Summary Complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - training_summary.json: Chi tiết metrics và checkpoints")
    if args.create_plots:
        print(f"  - training_plots.png: Biểu đồ training progress")
    print(f"\nLocation: {Path(args.checkpoints_dir).absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()

