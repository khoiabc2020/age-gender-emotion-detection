"""
Kiểm tra tổng thể toàn bộ dự án
Check tất cả các tuần từ 1-16
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

# Import check scripts
sys.path.insert(0, str(project_root / "ai_edge_app" / "scripts"))

def run_check(script_name, week_name):
    """Run check script"""
    try:
        script_path = project_root / "ai_edge_app" / "scripts" / script_name
        if script_path.exists():
            import subprocess
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            return result.returncode == 0, result.stdout, result.stderr
        else:
            return False, "", f"Script not found: {script_name}"
    except Exception as e:
        return False, "", str(e)

def main():
    """Main function"""
    print("\n" + "=" * 80)
    print(" KIỂM TRA TỔNG THỂ TOÀN BỘ DỰ ÁN")
    print("=" * 80)
    
    checks = [
        ("check_week1_requirements.py", "Data Processing"),
        ("check_week2_requirements.py", "Model Training"),
        ("check_week3_requirements.py", "Advanced Modules"),
        ("check_week4_requirements.py", "UI Framework"),
        ("check_week5_requirements.py", "Real-time Visualization"),
        ("check_week6_requirements.py", "Dynamic Ads System"),
        ("check_week7_requirements.py", "Business Logic & Tracking"),
        ("check_week8_requirements.py", "Multi-Threading"),
        ("check_week9_requirements.py", "Database & Reporting"),
    ]
    
    results = []
    
    for script, name in checks:
        print(f"\n{'='*80}")
        print(f" {name}")
        print('='*80)
        success, stdout, stderr = run_check(script, name)
        if stdout:
            print(stdout)
        if stderr and "Traceback" in stderr:
            print(f"⚠️  Error: {stderr[:200]}")
        results.append((name, success))
    
    # Summary
    print("\n" + "=" * 80)
    print(" TỔNG KẾT TẤT CẢ CÁC TUẦN")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nKết quả: {passed}/{total} tuần PASSED\n")
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:50s} {status}")
    
    print("\n" + "=" * 80)
    
    if passed == total:
        print(" TẤT CẢ CÁC TUẦN ĐÃ HOÀN THÀNH!")
    else:
        print(f"⚠️  {total - passed} tuần cần kiểm tra lại")
    
    print("=" * 80 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

