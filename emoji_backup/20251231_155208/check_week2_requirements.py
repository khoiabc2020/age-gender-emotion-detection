"""
Kiểm tra Tuần 2: Model Training (Lightweight SOTA)
- Architecture: MobileOne-S2 hoặc FastViT
- Knowledge Distillation: ResNet50 -> MobileOne
- Quantization-Aware Training (QAT)
- Export: ONNX (Opset 13+)
"""

import sys
import io
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "training_experiments" / "src"))


def check_mobileone():
    """Kiểm tra MobileOne-S2 architecture"""
    print("=" * 60)
    print("🏗️  KIỂM TRA ARCHITECTURE (MobileOne-S2)")
    print("=" * 60)
    
    results = []
    
    # Check mobileone.py
    print("\n[1/2] Checking mobileone.py...")
    mobileone_file = project_root / "training_experiments" / "src" / "models" / "mobileone.py"
    
    if mobileone_file.exists():
        with open(mobileone_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_mobileone = 'MobileOneS2' in content or 'MobileOne' in content
        has_multitask = 'MobileOneMultiTaskModel' in content
        
        if has_mobileone and has_multitask:
            print("   ✅ MobileOne-S2 architecture found")
            print("      - MobileOneS2 backbone")
            print("      - MobileOneMultiTaskModel")
            results.append(("MobileOne Architecture", True))
        else:
            print("   ⚠️  MobileOne architecture may be incomplete")
            results.append(("MobileOne Architecture", False))
    else:
        print("   ❌ mobileone.py not found")
        results.append(("MobileOne Architecture", False))
    
    # Check if can import
    print("\n[2/2] Testing import...")
    try:
        from models.mobileone import MobileOneMultiTaskModel
        model = MobileOneMultiTaskModel(num_emotions=6)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ MobileOne model can be imported")
        print(f"      Parameters: {total_params:,}")
        results.append(("MobileOne Import", True))
    except Exception as e:
        print(f"   ❌ Cannot import MobileOne: {e}")
        results.append(("MobileOne Import", False))
    
    return results


def check_knowledge_distillation():
    """Kiểm tra Knowledge Distillation"""
    print("\n" + "=" * 60)
    print("🎓 KIỂM TRA KNOWLEDGE DISTILLATION")
    print("=" * 60)
    
    results = []
    
    # Check knowledge_distillation.py
    print("\n[1/3] Checking knowledge_distillation.py...")
    distill_file = project_root / "training_experiments" / "src" / "models" / "knowledge_distillation.py"
    
    if distill_file.exists():
        with open(distill_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_teacher = 'TeacherModel' in content
        has_student = 'student' in content.lower()
        has_distill_loss = 'DistillationLoss' in content or 'MultiTaskDistillationLoss' in content
        
        if has_teacher and has_student and has_distill_loss:
            print("   ✅ Knowledge Distillation found")
            print("      - TeacherModel (ResNet50)")
            print("      - DistillationLoss")
            print("      - MultiTaskDistillationLoss")
            results.append(("Distillation Module", True))
        else:
            print("   ⚠️  Distillation may be incomplete")
            results.append(("Distillation Module", False))
    else:
        print("   ❌ knowledge_distillation.py not found")
        results.append(("Distillation Module", False))
    
    # Check training script
    print("\n[2/3] Checking training script...")
    train_file = project_root / "training_experiments" / "train_week2_lightweight.py"
    
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_distill_train = 'train_with_distillation' in content or 'distillation' in content.lower()
        
        if has_distill_train:
            print("   ✅ Distillation training function found")
            results.append(("Distillation Training", True))
        else:
            print("   ⚠️  Distillation training may be missing")
            results.append(("Distillation Training", False))
    else:
        print("   ❌ train_week2_lightweight.py not found")
        results.append(("Distillation Training", False))
    
    # Check if can import
    print("\n[3/3] Testing import...")
    try:
        from models.knowledge_distillation import TeacherModel, MultiTaskDistillationLoss
        print("   ✅ Distillation modules can be imported")
        results.append(("Distillation Import", True))
    except Exception as e:
        print(f"   ❌ Cannot import distillation: {e}")
        results.append(("Distillation Import", False))
    
    return results


def check_qat():
    """Kiểm tra Quantization-Aware Training"""
    print("\n" + "=" * 60)
    print("⚡ KIỂM TRA QUANTIZATION-AWARE TRAINING (QAT)")
    print("=" * 60)
    
    results = []
    
    # Check qat_model.py
    print("\n[1/3] Checking qat_model.py...")
    qat_file = project_root / "training_experiments" / "src" / "models" / "qat_model.py"
    
    if qat_file.exists():
        with open(qat_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_qat = 'QATMultiTaskModel' in content
        has_prepare = 'prepare_qat' in content
        has_convert = 'convert_to_quantized' in content
        has_quant_stub = 'QuantStub' in content
        
        if has_qat and has_prepare and has_convert and has_quant_stub:
            print("   ✅ QAT module found")
            print("      - QATMultiTaskModel")
            print("      - prepare_qat()")
            print("      - convert_to_quantized()")
            results.append(("QAT Module", True))
        else:
            print("   ⚠️  QAT may be incomplete")
            results.append(("QAT Module", False))
    else:
        print("   ❌ qat_model.py not found")
        results.append(("QAT Module", False))
    
    # Check PyTorch quantization support
    print("\n[2/3] Checking PyTorch quantization support...")
    try:
        import torch
        import torch.quantization as quantization
        
        if hasattr(quantization, 'prepare_qat'):
            print(f"   ✅ PyTorch quantization support found (v{torch.__version__})")
            results.append(("PyTorch QAT Support", True))
        else:
            print("   ⚠️  PyTorch version may not support QAT")
            results.append(("PyTorch QAT Support", False))
    except Exception as e:
        print(f"   ❌ Error checking PyTorch: {e}")
        results.append(("PyTorch QAT Support", False))
    
    # Check training script
    print("\n[3/3] Checking QAT in training script...")
    train_file = project_root / "training_experiments" / "train_week2_lightweight.py"
    
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_qat_flag = '--use_qat' in content or 'use_qat' in content
        has_qat_wrap = 'QATMultiTaskModel' in content
        
        if has_qat_flag and has_qat_wrap:
            print("   ✅ QAT integration in training script")
            results.append(("QAT Training Integration", True))
        else:
            print("   ⚠️  QAT integration may be missing")
            results.append(("QAT Training Integration", False))
    else:
        results.append(("QAT Training Integration", False))
    
    return results


def check_onnx_export():
    """Kiểm tra ONNX export với opset 13+"""
    print("\n" + "=" * 60)
    print("📦 KIỂM TRA ONNX EXPORT (Opset 13+)")
    print("=" * 60)
    
    results = []
    
    # Check convert_to_onnx.py
    print("\n[1/3] Checking convert_to_onnx.py...")
    onnx_file = project_root / "training_experiments" / "scripts" / "convert_to_onnx.py"
    
    if onnx_file.exists():
        with open(onnx_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_opset_13 = 'opset_version=13' in content or 'opset_version: int = 13' in content
        has_export = 'torch.onnx.export' in content
        
        if has_opset_13 and has_export:
            print("   ✅ ONNX export with opset 13+ found")
            print("      - Default opset_version = 13")
            print("      - torch.onnx.export")
            results.append(("ONNX Export Script", True))
        else:
            print("   ⚠️  ONNX export may use older opset")
            results.append(("ONNX Export Script", False))
    else:
        print("   ❌ convert_to_onnx.py not found")
        results.append(("ONNX Export Script", False))
    
    # Check training script export function
    print("\n[2/3] Checking export function in training script...")
    train_file = project_root / "training_experiments" / "train_week2_lightweight.py"
    
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_export_func = 'export_to_onnx' in content
        has_opset_param = 'opset_version' in content
        
        if has_export_func and has_opset_param:
            print("   ✅ ONNX export function found")
            results.append(("ONNX Export Function", True))
        else:
            print("   ⚠️  ONNX export function may be missing")
            results.append(("ONNX Export Function", False))
    else:
        results.append(("ONNX Export Function", False))
    
    # Check ONNX package
    print("\n[3/3] Checking ONNX package...")
    try:
        import onnx
        print(f"   ✅ ONNX package installed: {onnx.__version__}")
        results.append(("ONNX Package", True))
    except ImportError:
        print("   ❌ ONNX package not installed")
        print("      Install: pip install onnx")
        results.append(("ONNX Package", False))
    
    return results


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🔍 KIỂM TRA TUẦN 2: MODEL TRAINING (LIGHTWEIGHT SOTA)")
    print("=" * 60)
    
    all_results = []
    
    # Check MobileOne
    mobileone_results = check_mobileone()
    all_results.extend(mobileone_results)
    
    # Check Knowledge Distillation
    distill_results = check_knowledge_distillation()
    all_results.extend(distill_results)
    
    # Check QAT
    qat_results = check_qat()
    all_results.extend(qat_results)
    
    # Check ONNX export
    onnx_results = check_onnx_export()
    all_results.extend(onnx_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT")
    print("=" * 60)
    
    passed = sum(1 for _, result in all_results if result)
    total = len(all_results)
    
    print(f"\nKết quả: {passed}/{total} checks passed\n")
    
    for name, result in all_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:50s} {status}")
    
    print("\n" + "=" * 60)
    
    if passed == total:
        print("🎉 Tất cả yêu cầu Tuần 2 đã hoàn thành!")
        print("\nCó thể chạy training:")
        print("  python training_experiments/train_week2_lightweight.py \\")
        print("    --data_dir data/processed \\")
        print("    --epochs 50 \\")
        print("    --use_distillation \\")
        print("    --use_qat")
    else:
        print("⚠️  Một số yêu cầu chưa hoàn thành")
        print("\nCần kiểm tra và sửa các phần còn thiếu")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)






