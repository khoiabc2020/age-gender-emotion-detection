"""
Tự động cập nhật kết quả và đánh giá sau khi training xong
Chạy script này sau khi training hoàn thành để cập nhật tất cả kết quả
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# Fix encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


def check_training_status():
    """Kiểm tra trạng thái training"""
    results_dir = Path("results/auto_train_10x")
    
    if not results_dir.exists():
        return {'status': 'not_started', 'message': 'Training chưa bắt đầu'}
    
    summary_file = results_dir / "summary.json"
    if not summary_file.exists():
        return {'status': 'running', 'message': 'Training đang chạy...'}
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    total_runs = summary.get('total_runs', 0)
    successful_runs = summary.get('successful_runs', 0)
    
    if successful_runs == 0:
        return {'status': 'all_failed', 'message': 'Tất cả lần training đều thất bại'}
    elif successful_runs < total_runs:
        return {'status': 'partial', 'message': f'{successful_runs}/{total_runs} lần thành công'}
    else:
        return {'status': 'complete', 'message': 'Tất cả training đã hoàn thành'}


def analyze_and_update():
    """Phân tích và cập nhật kết quả"""
    print("\n" + "=" * 80)
    print("🔄 CẬP NHẬT KẾT QUẢ VÀ ĐÁNH GIÁ")
    print("=" * 80)
    
    # Check status
    status = check_training_status()
    print(f"\n📊 Trạng thái: {status['message']}")
    
    if status['status'] == 'not_started':
        print("❌ Training chưa bắt đầu. Chạy: python train_10x_automated.py")
        return
    
    if status['status'] == 'running':
        print("⏳ Training đang chạy. Vui lòng đợi...")
        return
    
    # Run analysis
    print("\n📈 Đang phân tích kết quả...")
    try:
        result = subprocess.run(
            [sys.executable, "analyze_results.py"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=60
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
    except Exception as e:
        print(f"❌ Lỗi khi phân tích: {e}")
    
    # Create final report
    create_final_report()


def create_final_report():
    """Tạo báo cáo cuối cùng"""
    results_dir = Path("results/auto_train_10x")
    summary_file = results_dir / "summary.json"
    
    if not summary_file.exists():
        return
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    report_file = results_dir / "FINAL_EVALUATION_REPORT.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 📊 BÁO CÁO ĐÁNH GIÁ CUỐI CÙNG - TRAINING 10 LẦN\n\n")
        f.write(f"**Ngày**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 📈 Tổng quan\n\n")
        f.write(f"- **Tổng số lần chạy**: {summary['total_runs']}\n")
        f.write(f"- **Số lần thành công**: {summary['successful_runs']}\n")
        f.write(f"- **Số lần thất bại**: {summary['total_runs'] - summary['successful_runs']}\n")
        f.write(f"- **Tỷ lệ thành công**: {summary['successful_runs'] / summary['total_runs'] * 100:.1f}%\n\n")
        
        if summary.get('best_run'):
            f.write("## 🏆 Best Run\n\n")
            best = summary['best_run']
            f.write(f"- **Run ID**: {best['run_id']}\n")
            f.write(f"- **Config**:\n")
            f.write(f"  ```json\n")
            f.write(f"  {json.dumps(best['config'], indent=2)}\n")
            f.write(f"  ```\n")
            f.write(f"- **Thời gian**: {best['elapsed_time']:.1f}s\n")
            if 'test_accuracy' in best:
                f.write(f"- **Test Accuracy**: {best['test_accuracy']:.4f}\n")
            f.write("\n")
        
        f.write("## 📋 Chi tiết từng Run\n\n")
        for i, result in enumerate(summary['results'], 1):
            f.write(f"### Run {i}\n\n")
            f.write(f"- **Config**: {json.dumps(result['config'], indent=2)}\n")
            f.write(f"- **Status**: {'✅ Success' if result['success'] else '❌ Failed'}\n")
            f.write(f"- **Thời gian**: {result['elapsed_time']:.1f}s\n")
            if result.get('stdout'):
                f.write(f"- **Error Output**: {result['stdout'][:200]}...\n")
            f.write("\n")
        
        f.write("---\n\n")
        f.write("## 💡 Kết luận\n\n")
        
        if summary['successful_runs'] == 0:
            f.write("⚠️ **Tất cả lần training đều thất bại.**\n\n")
            f.write("**Nguyên nhân có thể:**\n")
            f.write("1. Thiếu dữ liệu training\n")
            f.write("2. Lỗi trong script training\n")
            f.write("3. Thiếu dependencies\n")
            f.write("4. Lỗi cấu hình\n\n")
            f.write("**Giải pháp:**\n")
            f.write("1. Kiểm tra dữ liệu: `python scripts/check_datasets.py`\n")
            f.write("2. Kiểm tra log: Xem `results/auto_train_10x/run_*_results.json`\n")
            f.write("3. Chạy thử 1 lần: `python train_week2_lightweight.py --data_dir data/processed --epochs 1`\n")
        elif summary['successful_runs'] > 0:
            f.write(f"✅ **{summary['successful_runs']} lần training thành công!**\n\n")
            if summary.get('best_run'):
                f.write(f"**Best Model**: Run {summary['best_run']['run_id']}\n")
                f.write(f"**Location**: `results/auto_train_10x/run_{summary['best_run']['run_id']}/best_model.pth`\n")
        f.write("\n")
        
        f.write("---\n\n")
        f.write("**Status**: ✅ Report Complete\n")
    
    print(f"\n✅ Báo cáo đã lưu vào: {report_file}")


if __name__ == "__main__":
    analyze_and_update()






