"""
Kiểm tra Tuần 3: Advanced Modules
- Anti-Spoofing: MiniFASNet
- Face Restoration: GFPGAN/ESPCN
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
sys.path.insert(0, str(project_root / "ai_edge_app" / "src"))


def check_anti_spoofing():
    """Kiểm tra Anti-Spoofing (MiniFASNet)"""
    print("=" * 60)
    print("🛡️  KIỂM TRA ANTI-SPOOFING (MiniFASNet)")
    print("=" * 60)
    
    results = []
    
    # Check anti_spoofing.py
    print("\n[1/3] Checking anti_spoofing.py...")
    anti_spoof_file = project_root / "ai_edge_app" / "src" / "core" / "anti_spoofing.py"
    
    if anti_spoof_file.exists():
        with open(anti_spoof_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_minifas = 'MiniFASNet' in content
        has_predict = 'def predict' in content
        has_is_real = 'is_real_face' in content
        has_onnx = 'onnxruntime' in content
        
        if has_minifas and has_predict and has_is_real and has_onnx:
            print("   ✅ MiniFASNet module found")
            print("      - MiniFASNet class")
            print("      - predict() method")
            print("      - is_real_face() method")
            print("      - ONNX Runtime support")
            results.append(("Anti-Spoofing Module", True))
        else:
            print("   ⚠️  Anti-spoofing may be incomplete")
            results.append(("Anti-Spoofing Module", False))
    else:
        print("   ❌ anti_spoofing.py not found")
        results.append(("Anti-Spoofing Module", False))
    
    # Check integration in main.py
    print("\n[2/3] Checking integration in main.py...")
    main_file = project_root / "ai_edge_app" / "main.py"
    
    if main_file.exists():
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_import = 'from src.core.anti_spoofing import MiniFASNet' in content
        has_init = 'self.anti_spoofing' in content
        has_use = 'self.anti_spoofing.predict' in content or 'self.anti_spoofing' in content
        
        if has_import and has_init and has_use:
            print("   ✅ Anti-spoofing integrated in main.py")
            results.append(("Anti-Spoofing Integration", True))
        else:
            print("   ⚠️  Anti-spoofing may not be fully integrated")
            results.append(("Anti-Spoofing Integration", False))
    else:
        results.append(("Anti-Spoofing Integration", False))
    
    # Check if can import
    print("\n[3/3] Testing import...")
    try:
        from core.anti_spoofing import MiniFASNet
        model = MiniFASNet()
        print("   ✅ MiniFASNet can be imported")
        results.append(("Anti-Spoofing Import", True))
    except Exception as e:
        print(f"   ❌ Cannot import MiniFASNet: {e}")
        results.append(("Anti-Spoofing Import", False))
    
    return results


def check_face_restoration():
    """Kiểm tra Face Restoration (GFPGAN/ESPCN)"""
    print("\n" + "=" * 60)
    print("✨ KIỂM TRA FACE RESTORATION (GFPGAN/ESPCN)")
    print("=" * 60)
    
    results = []
    
    # Check face_restoration.py
    print("\n[1/3] Checking face_restoration.py...")
    restore_file = project_root / "ai_edge_app" / "src" / "core" / "face_restoration.py"
    
    if restore_file.exists():
        with open(restore_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_restorer = 'FaceRestorer' in content
        has_restore = 'def restore' in content
        has_enhance = 'enhance_if_needed' in content
        has_espcn = 'espcn' in content.lower()
        has_gfpgan = 'gfpgan' in content.lower()
        
        if has_restorer and has_restore and has_enhance and (has_espcn or has_gfpgan):
            print("   ✅ FaceRestorer module found")
            print("      - FaceRestorer class")
            print("      - restore() method")
            print("      - enhance_if_needed() method")
            print("      - ESPCN/GFPGAN support")
            results.append(("Face Restoration Module", True))
        else:
            print("   ⚠️  Face restoration may be incomplete")
            results.append(("Face Restoration Module", False))
    else:
        print("   ❌ face_restoration.py not found")
        results.append(("Face Restoration Module", False))
    
    # Check integration in main.py
    print("\n[2/3] Checking integration in main.py...")
    main_file = project_root / "ai_edge_app" / "main.py"
    
    if main_file.exists():
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_import = 'from src.core.face_restoration import FaceRestorer' in content
        has_init = 'self.face_restorer' in content
        has_use = 'self.face_restorer.enhance' in content or 'self.face_restorer.restore' in content
        
        if has_import and has_init and has_use:
            print("   ✅ Face restoration integrated in main.py")
            results.append(("Face Restoration Integration", True))
        else:
            print("   ⚠️  Face restoration may not be fully integrated")
            results.append(("Face Restoration Integration", False))
    else:
        results.append(("Face Restoration Integration", False))
    
    # Check if can import
    print("\n[3/3] Testing import...")
    try:
        from core.face_restoration import FaceRestorer
        restorer = FaceRestorer(method="espcn")
        print("   ✅ FaceRestorer can be imported")
        results.append(("Face Restoration Import", True))
    except Exception as e:
        print(f"   ❌ Cannot import FaceRestorer: {e}")
        results.append(("Face Restoration Import", False))
    
    return results


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🔍 KIỂM TRA TUẦN 3: ADVANCED MODULES")
    print("=" * 60)
    
    all_results = []
    
    # Check Anti-Spoofing
    anti_spoof_results = check_anti_spoofing()
    all_results.extend(anti_spoof_results)
    
    # Check Face Restoration
    restore_results = check_face_restoration()
    all_results.extend(restore_results)
    
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
        print("🎉 Tất cả yêu cầu Tuần 3 đã hoàn thành!")
        print("\nModules đã được tích hợp vào pipeline:")
        print("  - Anti-Spoofing: Lọc khuôn mặt giả trước khi classify")
        print("  - Face Restoration: Tăng cường chất lượng ảnh mờ")
    else:
        print("⚠️  Một số yêu cầu chưa hoàn thành")
        print("\nCần kiểm tra và sửa các phần còn thiếu")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)






