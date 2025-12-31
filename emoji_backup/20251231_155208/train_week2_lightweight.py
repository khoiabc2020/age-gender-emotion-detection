"""
Tuần 2: Model Training (Lightweight SOTA)
- Architecture: MobileOne-S2
- Knowledge Distillation: ResNet50 -> MobileOne
- Quantization-Aware Training (QAT)
- Export: ONNX (Opset 13+)
"""

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import sys
import io

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from src.models.mobileone import MobileOneMultiTaskModel
from src.models.knowledge_distillation import TeacherModel, MultiTaskDistillationLoss
from src.models.qat_model import QATMultiTaskModel
from src.data.dataset import get_dataloaders


def train_with_distillation(
    student_model,
    teacher_model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    use_mixed_precision=True,
    scaler=None,
    max_grad_norm=1.0
):
    """Train student model with knowledge distillation"""
    student_model.train()
    teacher_model.eval()  # Teacher is frozen
    
    running_loss = {
        'total': 0.0, 'gender': 0.0, 'age': 0.0, 'emotion': 0.0,
        'gender_soft': 0.0, 'gender_hard': 0.0,
        'emotion_soft': 0.0, 'emotion_hard': 0.0
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Distillation]')
    
    for batch in pbar:
        images = batch['image'].to(device)
        gender_target = batch['gender'].to(device)
        age_target = batch['age'].to(device)
        emotion_target = batch['emotion'].to(device)
        
        optimizer.zero_grad()
        
        # Get teacher predictions (no gradients)
        with torch.no_grad():
            teacher_gender, teacher_age, teacher_emotion = teacher_model(images)
        
        # Get student predictions
        if use_mixed_precision and scaler is not None:
            with autocast():
                student_gender, student_age, student_emotion = student_model(images)
                
                student_preds = (student_gender, student_age, student_emotion)
                teacher_preds = (teacher_gender, teacher_age, teacher_emotion)
                targets = (gender_target, age_target, emotion_target)
                
                losses = criterion(student_preds, teacher_preds, targets)
            
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            student_gender, student_age, student_emotion = student_model(images)
            
            student_preds = (student_gender, student_age, student_emotion)
            teacher_preds = (teacher_gender, teacher_age, teacher_emotion)
            targets = (gender_target, age_target, emotion_target)
            
            losses = criterion(student_preds, teacher_preds, targets)
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
            optimizer.step()
        
        # Update running loss
        for key in running_loss:
            if key in losses:
                running_loss[key] += losses[key].item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{losses['total'].item():.4f}",
            'G': f"{losses['gender'].item():.4f}",
            'A': f"{losses['age'].item():.4f}",
            'E': f"{losses['emotion'].item():.4f}"
        })
    
    # Average losses
    for key in running_loss:
        running_loss[key] /= len(train_loader)
    
    return running_loss


def validate(model, val_loader, device):
    """Validate model"""
    model.eval()
    total_gender_correct = 0
    total_age_error = 0.0
    total_emotion_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            images = batch['image'].to(device)
            gender_target = batch['gender'].to(device)
            age_target = batch['age'].to(device)
            emotion_target = batch['emotion'].to(device)
            
            gender_logits, age_pred, emotion_logits = model(images)
            
            # Gender accuracy
            gender_pred = gender_logits.argmax(dim=1)
            total_gender_correct += (gender_pred == gender_target).sum().item()
            
            # Age MAE
            age_error = torch.abs(age_pred.squeeze() - age_target.float())
            total_age_error += age_error.sum().item()
            
            # Emotion accuracy
            emotion_pred = emotion_logits.argmax(dim=1)
            total_emotion_correct += (emotion_pred == emotion_target).sum().item()
            
            total_samples += images.size(0)
    
    metrics = {
        'gender_acc': total_gender_correct / total_samples,
        'age_mae': total_age_error / total_samples,
        'emotion_acc': total_emotion_correct / total_samples
    }
    
    return metrics


def export_to_onnx(model, output_path, opset_version=13):
    """Export model to ONNX with opset 13+"""
    print(f"\nExporting to ONNX (opset {opset_version})...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['gender', 'age', 'emotion'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'gender': {0: 'batch_size'},
            'age': {0: 'batch_size'},
            'emotion': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"✅ Model exported to: {output_path}")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model valid (opset {onnx_model.opset_import[0].version})")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify ONNX model: {e}")


def main():
    parser = argparse.ArgumentParser(description='Tuần 2: Lightweight SOTA Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints/week2', help='Save directory')
    parser.add_argument('--use_qat', action='store_true', help='Use Quantization-Aware Training')
    parser.add_argument('--use_distillation', action='store_true', default=True, help='Use Knowledge Distillation')
    parser.add_argument('--temperature', type=float, default=4.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='Distillation alpha')
    parser.add_argument('--opset_version', type=int, default=13, help='ONNX opset version')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers (0 for Windows compatibility)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Teacher model (ResNet50)
    if args.use_distillation:
        print("\nLoading teacher model (ResNet50)...")
        teacher_model = TeacherModel(num_emotions=6).to(device)
        teacher_model.eval()  # Freeze teacher
        for param in teacher_model.parameters():
            param.requires_grad = False
        print("✅ Teacher model loaded")
    else:
        teacher_model = None
    
    # Student model (MobileOne-S2)
    print("\nCreating student model (MobileOne-S2)...")
    base_student = MobileOneMultiTaskModel(num_emotions=6, width_multiplier=1.0).to(device)
    
    # Wrap with QAT if enabled
    if args.use_qat:
        print("Enabling Quantization-Aware Training...")
        student_model = QATMultiTaskModel(base_student).to(device)
        student_model.prepare_qat()
        print("✅ QAT enabled")
    else:
        student_model = base_student
    
    total_params = sum(p.numel() for p in student_model.parameters())
    print(f"✅ Student model created: {total_params:,} parameters")
    
    # Loss function
    if args.use_distillation:
        criterion = MultiTaskDistillationLoss(
            temperature=args.temperature,
            alpha=args.alpha
        )
    else:
        from src.models.network import MultiTaskLoss
        criterion = MultiTaskLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Mixed precision
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=save_dir / 'logs')
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training (Tuần 2: Lightweight SOTA)")
    print("=" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train
        if args.use_distillation:
            train_losses = train_with_distillation(
                student_model, teacher_model, train_loader,
                criterion, optimizer, device, epoch,
                use_mixed_precision=(device.type == 'cuda'),
                scaler=scaler
            )
        else:
            # Standard training (implement if needed)
            pass
        
        # Validate
        val_metrics = validate(student_model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Train/Loss', train_losses['total'], epoch)
        writer.add_scalar('Val/Gender_Acc', val_metrics['gender_acc'], epoch)
        writer.add_scalar('Val/Age_MAE', val_metrics['age_mae'], epoch)
        writer.add_scalar('Val/Emotion_Acc', val_metrics['emotion_acc'], epoch)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"  Val - Gender Acc: {val_metrics['gender_acc']:.4f}, "
              f"Age MAE: {val_metrics['age_mae']:.2f}, "
              f"Emotion Acc: {val_metrics['emotion_acc']:.4f}")
        
        # Save best model
        current_acc = val_metrics['emotion_acc']
        if current_acc > best_val_acc:
            best_val_acc = current_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'best_val_acc': best_val_acc
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f"  ✅ Saved best model (Acc: {best_val_acc:.4f})")
    
    # Export to ONNX
    print("\n" + "=" * 60)
    print("Exporting to ONNX...")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(save_dir / 'best_model.pth', map_location=device)
    student_model.load_state_dict(checkpoint['model_state_dict'])
    student_model.eval()
    
    # If QAT, convert to quantized
    if args.use_qat:
        print("Converting QAT model to quantized...")
        student_model = student_model.convert_to_quantized()
    
    # Export
    onnx_path = save_dir / 'mobileone_multitask.onnx'
    export_to_onnx(student_model, onnx_path, opset_version=args.opset_version)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"Best Model: {save_dir / 'best_model.pth'}")
    print(f"ONNX Model: {onnx_path}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()


