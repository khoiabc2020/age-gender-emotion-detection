"""
Script tự động commit và push code lên GitHub
Có thể chạy định kỳ hoặc sau mỗi thay đổi
"""

import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime

def run_command(cmd, cwd=None):
    """Chạy lệnh shell và trả về kết quả"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def check_git_repo():
    """Kiểm tra xem có phải git repo không"""
    project_root = Path(__file__).parent.parent.parent
    git_dir = project_root / '.git'
    return git_dir.exists()


def get_git_status():
    """Lấy trạng thái git"""
    project_root = Path(__file__).parent.parent.parent
    success, stdout, stderr = run_command('git status --porcelain', cwd=project_root)
    return success, stdout.strip()


def auto_commit_push():
    """Tự động commit và push"""
    project_root = Path(__file__).parent.parent.parent
    
    # Kiểm tra git repo
    if not check_git_repo():
        print("❌ Không phải git repository!")
        print("   Chạy: git init")
        return False
    
    # Kiểm tra có thay đổi không
    success, status = get_git_status()
    if not success:
        print("❌ Lỗi khi kiểm tra git status")
        return False
    
    if not status:
        print("✅ Không có thay đổi nào để commit")
        return True
    
    print("📝 Phát hiện thay đổi:")
    print(status)
    print()
    
    # Add tất cả thay đổi
    print("📦 Đang add files...")
    success, stdout, stderr = run_command('git add .', cwd=project_root)
    if not success:
        print(f"❌ Lỗi khi add files: {stderr}")
        return False
    
    # Commit
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    commit_message = f"Auto commit: {timestamp}"
    
    print(f"💾 Đang commit: {commit_message}")
    success, stdout, stderr = run_command(
        f'git commit -m "{commit_message}"',
        cwd=project_root
    )
    
    if not success:
        if "nothing to commit" in stderr.lower():
            print("✅ Không có gì để commit")
            return True
        print(f"❌ Lỗi khi commit: {stderr}")
        return False
    
    print("✅ Đã commit thành công")
    
    # Push
    print("📤 Đang push lên GitHub...")
    success, stdout, stderr = run_command('git push', cwd=project_root)
    
    if not success:
        if "no upstream branch" in stderr.lower():
            print("⚠️  Chưa có upstream branch")
            print("   Chạy: git push -u origin main")
            return False
        print(f"❌ Lỗi khi push: {stderr}")
        return False
    
    print("✅ Đã push lên GitHub thành công!")
    return True


def setup_git_repo():
    """Setup git repo nếu chưa có"""
    project_root = Path(__file__).parent.parent.parent
    
    if check_git_repo():
        print("✅ Đã là git repository")
        return True
    
    print("🔧 Đang setup git repository...")
    
    # Init git
    success, stdout, stderr = run_command('git init', cwd=project_root)
    if not success:
        print(f"❌ Lỗi khi init git: {stderr}")
        return False
    
    # Tạo .gitignore nếu chưa có
    gitignore = project_root / '.gitignore'
    if not gitignore.exists():
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
venv_gpu/
env/
ENV/

# Data
data/
*.zip
*.pth
*.onnx
*.h5
*.ckpt

# Logs
logs/
*.log

# Checkpoints
checkpoints/
results/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Temporary
*.tmp
*.temp
"""
        gitignore.write_text(gitignore_content, encoding='utf-8')
        print("✅ Đã tạo .gitignore")
    
    print("✅ Đã setup git repository")
    print("\n📝 Các bước tiếp theo:")
    print("1. Thêm remote: git remote add origin <your-github-repo-url>")
    print("2. Commit lần đầu: git add . && git commit -m 'Initial commit'")
    print("3. Push: git push -u origin main")
    
    return True


def main():
    """Hàm chính"""
    try:
        print("=" * 60)
        print("Tu dong Commit va Push len GitHub")
        print("=" * 60)
        print()
    except:
        print("=" * 60)
        print("Auto Commit and Push to GitHub")
        print("=" * 60)
        print()
    
    # Kiểm tra và setup git nếu cần
    if not check_git_repo():
        setup_git_repo()
        return
    
    # Tự động commit và push
    success = auto_commit_push()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Hoàn tất!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Có lỗi xảy ra")
        print("=" * 60)


if __name__ == "__main__":
    main()

