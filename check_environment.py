"""
Script kiểm tra môi trường để chạy Smart Retail AI
"""

import sys
import subprocess
import os
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def check_command(cmd, version_flag='--version'):
    """Kiểm tra command có tồn tại không"""
    try:
        result = subprocess.run(
            [cmd, version_flag],
            capture_output=True,
            text=True,
            timeout=5,
            shell=True
        )
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            return True, version
        return False, None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, None

def check_python_package(package):
    """Kiểm tra Python package đã cài chưa"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def check_file_exists(filepath):
    """Kiểm tra file có tồn tại không"""
    return Path(filepath).exists()

def check_directory_exists(dirpath):
    """Kiểm tra thư mục có tồn tại không"""
    return Path(dirpath).is_dir()

def main():
    print_header("KIỂM TRA MÔI TRƯỜNG - SMART RETAIL AI")
    
    issues = []
    warnings = []
    
    # 1. Kiểm tra Python
    print("\n[1] Kiểm tra Python...")
    python_ok, python_version = check_command("python")
    if python_ok:
        print_success(f"Python: {python_version}")
        # Kiểm tra version >= 3.10
        try:
            version_num = float(python_version.split()[1][:3])
            if version_num < 3.10:
                print_warning(f"Python version {version_num} < 3.10 (khuyến nghị 3.10+)")
                warnings.append("Python version < 3.10")
        except:
            pass
    else:
        print_error("Python không tìm thấy!")
        issues.append("Python chưa cài đặt")
    
    # 2. Kiểm tra Node.js
    print("\n[2] Kiểm tra Node.js...")
    node_ok, node_version = check_command("node")
    if node_ok:
        print_success(f"Node.js: {node_version}")
        # Kiểm tra version >= 18
        try:
            version_num = int(node_version.split('v')[1].split('.')[0])
            if version_num < 18:
                print_warning(f"Node.js version {version_num} < 18 (khuyến nghị 18+)")
                warnings.append("Node.js version < 18")
        except:
            pass
    else:
        print_error("Node.js không tìm thấy!")
        issues.append("Node.js chưa cài đặt")
    
    # 3. Kiểm tra npm
    print("\n[3] Kiểm tra npm...")
    npm_ok, npm_version = check_command("npm")
    if npm_ok:
        print_success(f"npm: {npm_version}")
    else:
        # Thử kiểm tra node_modules thay vì npm
        if check_directory_exists(project_root / "dashboard" / "node_modules"):
            print_success("npm đã được sử dụng (node_modules tồn tại)")
        else:
            print_warning("npm không tìm thấy (nhưng có thể dùng npx hoặc yarn)")
            warnings.append("npm không tìm thấy trong PATH")
    
    # 4. Kiểm tra pip
    print("\n[4] Kiểm tra pip...")
    pip_ok, pip_version = check_command("pip")
    if pip_ok:
        print_success(f"pip: {pip_version}")
    else:
        print_error("pip không tìm thấy!")
        issues.append("pip chưa cài đặt")
    
    # 5. Kiểm tra thư mục dự án
    print("\n[5] Kiểm tra cấu trúc dự án...")
    project_root = Path(__file__).parent
    
    required_dirs = [
        "backend_api",
        "dashboard",
        "ai_edge_app",
        "training_experiments"
    ]
    
    for dir_name in required_dirs:
        if check_directory_exists(project_root / dir_name):
            print_success(f"Thư mục: {dir_name}/")
        else:
            print_error(f"Thiếu thư mục: {dir_name}/")
            issues.append(f"Thiếu thư mục: {dir_name}/")
    
    # 6. Kiểm tra Backend
    print("\n[6] Kiểm tra Backend...")
    backend_dir = project_root / "backend_api"
    
    # requirements.txt
    if check_file_exists(backend_dir / "requirements.txt"):
        print_success("requirements.txt tồn tại")
    else:
        print_error("Thiếu: backend_api/requirements.txt")
        issues.append("Thiếu backend_api/requirements.txt")
    
    # .env
    if check_file_exists(backend_dir / ".env"):
        print_success(".env tồn tại")
    else:
        print_warning(".env chưa có (sẽ tự động tạo khi chạy)")
        warnings.append("backend_api/.env chưa có")
    
    # venv
    if check_directory_exists(backend_dir / "venv"):
        print_success("venv tồn tại")
    else:
        print_warning("venv chưa có (sẽ tự động tạo khi chạy)")
        warnings.append("backend_api/venv chưa có")
    
    # 7. Kiểm tra Frontend
    print("\n[7] Kiểm tra Frontend...")
    dashboard_dir = project_root / "dashboard"
    
    # package.json
    if check_file_exists(dashboard_dir / "package.json"):
        print_success("package.json tồn tại")
    else:
        print_error("Thiếu: dashboard/package.json")
        issues.append("Thiếu dashboard/package.json")
    
    # node_modules
    if check_directory_exists(dashboard_dir / "node_modules"):
        print_success("node_modules tồn tại")
    else:
        print_warning("node_modules chưa có (cần chạy: npm install)")
        warnings.append("dashboard/node_modules chưa có")
    
    # .env.local
    if check_file_exists(dashboard_dir / ".env.local"):
        print_success(".env.local tồn tại")
    else:
        print_warning(".env.local chưa có (sẽ tự động tạo khi chạy)")
        warnings.append("dashboard/.env.local chưa có")
    
    # 8. Kiểm tra Python packages quan trọng
    print("\n[8] Kiểm tra Python packages...")
    
    # Kiểm tra trong venv nếu có
    venv_python = project_root / "backend_api" / "venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        print(f"  Kiểm tra trong venv: {venv_python}")
        import subprocess
        try:
            result = subprocess.run(
                [str(venv_python), "-c", "import fastapi; print(fastapi.__version__)"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print_success(f"FastAPI đã cài trong venv: {result.stdout.strip()}")
            else:
                print_warning("FastAPI chưa cài trong venv")
                warnings.append("FastAPI chưa cài trong venv")
        except:
            print_warning("Không thể kiểm tra packages trong venv")
            warnings.append("Không thể kiểm tra venv packages")
    else:
        # Kiểm tra global
        important_packages = [
            ("fastapi", "FastAPI"),
            ("uvicorn", "Uvicorn"),
            ("torch", "PyTorch"),
            ("numpy", "NumPy"),
            ("cv2", "OpenCV"),
        ]
        
        for package, name in important_packages:
            if check_python_package(package):
                print_success(f"{name} đã cài")
            else:
                print_warning(f"{name} chưa cài (sẽ cài khi chạy run_backend.bat)")
                warnings.append(f"{name} chưa cài")
    
    # 9. Kiểm tra scripts
    print("\n[9] Kiểm tra Scripts...")
    required_scripts = [
        "START_PROJECT.bat",
        "run_backend.bat",
        "run_frontend.bat",
        "run_training_test.bat"
    ]
    
    for script in required_scripts:
        if check_file_exists(project_root / script):
            print_success(f"{script} tồn tại")
        else:
            print_error(f"Thiếu: {script}")
            issues.append(f"Thiếu {script}")
    
    # 10. Kiểm tra ports
    print("\n[10] Kiểm tra Ports...")
    import socket
    
    ports_to_check = [
        (8000, "Backend API"),
        (3000, "Frontend Dashboard"),
    ]
    
    for port, name in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result == 0:
            print_warning(f"Port {port} ({name}) đang được sử dụng")
            warnings.append(f"Port {port} đang được sử dụng")
        else:
            print_success(f"Port {port} ({name}) trống")
    
    # Tổng kết
    print_header("TỔNG KẾT")
    
    if not issues:
        print_success("Không có lỗi nghiêm trọng!")
    else:
        print_error(f"Có {len(issues)} vấn đề cần xử lý:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    if warnings:
        print_warning(f"Có {len(warnings)} cảnh báo:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    print("\n" + "="*60)
    if not issues:
        print(f"{GREEN}✓ MÔI TRƯỜNG SẴN SÀNG!{RESET}")
        print("\nĐể chạy dự án:")
        print("  1. Chạy: START_PROJECT.bat")
        print("  2. Chọn option 4 (All Services)")
        print("  3. Truy cập: http://localhost:3000")
    else:
        print(f"{RED}✗ CẦN XỬ LÝ CÁC VẤN ĐỀ TRƯỚC!{RESET}")
    print("="*60 + "\n")
    
    return len(issues) == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nĐã hủy kiểm tra.")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Lỗi: {e}{RESET}")
        sys.exit(1)

