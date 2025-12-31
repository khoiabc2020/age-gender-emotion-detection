"""
Script tự động chuẩn bị dữ liệu và chạy training
Kiểm tra và chuẩn bị data trước khi training
"""

import sys
import subprocess
from pathlib import Path

# Fix encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


def check_data():
    """Kiểm tra dữ liệu đã được preprocess chưa"""
    data_dir = Path("data/processed")
    
    if not data_dir.exists():
        return False, "Thư mục data/processed không tồn tại"
    
    # Kiểm tra có train/val/test splits không
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    
    if not train_dir.exists():
        return False, "Thư mục train không tồn tại"
    
    # Kiểm tra có ảnh không
    train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    if len(train_images) == 0:
        # Kiểm tra cấu trúc theo class
        subdirs = [d for d in train_dir.iterdir() if d.is_dir()]
        if subdirs:
            total_images = 0
            for subdir in subdirs:
                images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
                total_images += len(images)
            if total_images == 0:
                return False, "Không tìm thấy ảnh trong data/processed/train"
        else:
            return False, "Không tìm thấy ảnh trong data/processed/train"
    
    return True, "Dữ liệu đã sẵn sàng"


def prepare_data():
    """Chuẩn bị dữ liệu nếu chưa có"""
    print("\n" + "=" * 80)
    print("📦 CHUẨN BỊ DỮ LIỆU")
    print("=" * 80)
    
    # Kiểm tra data
    has_data, message = check_data()
    
    if has_data:
        print(f"✅ {message}")
        return True
    
    print(f"❌ {message}")
    print("\n🔧 Đang chuẩn bị dữ liệu...")
    
    # Chạy script preprocess
    try:
        print("\n[1/3] Kiểm tra datasets...")
        result = subprocess.run(
            [sys.executable, "scripts/check_datasets.py"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300
        )
        print(result.stdout)
        
        print("\n[2/3] Preprocessing data...")
        # Có thể cần chạy preprocess script
        # Tạm thời chỉ báo lỗi
        print("⚠️ Cần chạy preprocessing script trước!")
        print("   Chạy: python scripts/copy_datasets_to_project.py")
        print("   Sau đó: python src/data/preprocess.py")
        
        return False
        
    except Exception as e:
        print(f"❌ Lỗi khi chuẩn bị dữ liệu: {e}")
        return False


def run_training():
    """Chạy training"""
    print("\n" + "=" * 80)
    print("🚀 BẮT ĐẦU TRAINING")
    print("=" * 80)
    
    # Kiểm tra lại data
    has_data, message = check_data()
    if not has_data:
        print(f"❌ {message}")
        print("\n⚠️ Không thể chạy training vì thiếu dữ liệu!")
        print("\n📋 Hướng dẫn chuẩn bị dữ liệu:")
        print("   1. Download datasets từ Kaggle")
        print("   2. Copy vào project: python scripts/copy_datasets_to_project.py")
        print("   3. Preprocess: python src/data/preprocess.py")
        return False
    
    # Chạy training
    print(f"✅ {message}")
    print("\n🚀 Đang chạy training 10 lần...")
    
    try:
        result = subprocess.run(
            [sys.executable, "train_10x_automated.py"],
            cwd=Path(__file__).parent,
            timeout=3600 * 24  # 24 hours timeout
        )
        
        if result.returncode == 0:
            print("\n✅ Training hoàn thành!")
            return True
        else:
            print(f"\n❌ Training thất bại với returncode: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n⏰ Training timeout (quá 24 giờ)")
        return False
    except Exception as e:
        print(f"\n❌ Lỗi khi chạy training: {e}")
        return False


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("🎯 CHUẨN BỊ & TRAINING TỰ ĐỘNG")
    print("=" * 80)
    
    # Step 1: Prepare data
    if not prepare_data():
        print("\n❌ Không thể chuẩn bị dữ liệu. Vui lòng chuẩn bị thủ công.")
        return
    
    # Step 2: Run training
    success = run_training()
    
    if success:
        print("\n" + "=" * 80)
        print("✅ HOÀN THÀNH!")
        print("=" * 80)
        print("\n📊 Phân tích kết quả:")
        print("   python analyze_results.py")
        print("   python update_results_and_evaluate.py")
    else:
        print("\n" + "=" * 80)
        print("❌ TRAINING THẤT BẠI")
        print("=" * 80)
        print("\n📋 Kiểm tra:")
        print("   1. Xem log: results/auto_train_10x/run_*_results.json")
        print("   2. Kiểm tra data: python scripts/check_datasets.py")
        print("   3. Chạy thử 1 lần: python train_week2_lightweight.py --data_dir data/processed --epochs 1")


if __name__ == "__main__":
    main()






