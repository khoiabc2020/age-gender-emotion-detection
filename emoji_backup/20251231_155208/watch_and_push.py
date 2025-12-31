"""
Script theo dõi thay đổi file và tự động push lên GitHub
Chạy script này để tự động sync code lên GitHub khi có thay đổi
"""

import time
import subprocess
import os
from pathlib import Path
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️  Cần cài đặt: pip install watchdog")


class GitAutoPushHandler(FileSystemEventHandler):
    """Handler tự động commit và push khi có thay đổi"""
    
    def __init__(self, project_root, delay=30):
        self.project_root = Path(project_root)
        self.delay = delay  # Đợi delay giây trước khi commit (tránh commit quá nhiều)
        self.last_commit_time = 0
        self.pending_changes = False
    
    def on_modified(self, event):
        """Khi file được sửa đổi"""
        if event.is_directory:
            return
        
        # Bỏ qua các file không cần thiết
        ignored_extensions = {'.pyc', '.pyo', '.pyd', '.log', '.tmp', '.swp'}
        if Path(event.src_path).suffix in ignored_extensions:
            return
        
        # Bỏ qua thư mục không cần thiết
        ignored_dirs = {'__pycache__', '.git', 'venv', 'venv_gpu', 'node_modules', 
                       'checkpoints', 'results', 'logs', '.ipynb_checkpoints'}
        if any(ignored in event.src_path for ignored in ignored_dirs):
            return
        
        print(f"📝 Phát hiện thay đổi: {event.src_path}")
        self.pending_changes = True
        
        # Đợi delay giây trước khi commit (tránh commit quá nhiều)
        current_time = time.time()
        if current_time - self.last_commit_time > self.delay:
            self.commit_and_push()
            self.last_commit_time = current_time
            self.pending_changes = False
    
    def commit_and_push(self):
        """Commit và push lên GitHub"""
        print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Đang commit và push...")
        
        # Add
        subprocess.run(['git', 'add', '.'], cwd=self.project_root, 
                      capture_output=True)
        
        # Commit
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        commit_msg = f"Auto commit: {timestamp}"
        result = subprocess.run(
            ['git', 'commit', '-m', commit_msg],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and "nothing to commit" not in result.stdout.lower():
            print(f"✅ Đã commit: {commit_msg}")
            
            # Push
            push_result = subprocess.run(
                ['git', 'push'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if push_result.returncode == 0:
                print(f"✅ Đã push lên GitHub")
            else:
                print(f"⚠️  Lỗi khi push: {push_result.stderr}")
        else:
            print("ℹ️  Không có gì để commit")


def watch_and_push(project_root=None, delay=30):
    """Theo dõi thay đổi và tự động push"""
    if not WATCHDOG_AVAILABLE:
        print("❌ Cần cài đặt watchdog: pip install watchdog")
        return
    
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    project_root = Path(project_root)
    
    if not (project_root / '.git').exists():
        print("❌ Không phải git repository!")
        print("   Chạy: python scripts/auto_git_push.py để setup")
        return
    
    print("=" * 60)
    print("👀 Đang theo dõi thay đổi file...")
    print(f"📁 Thư mục: {project_root}")
    print(f"⏱️  Delay: {delay} giây")
    print("=" * 60)
    print("💡 Nhấn Ctrl+C để dừng")
    print()
    
    event_handler = GitAutoPushHandler(project_root, delay)
    observer = Observer()
    observer.schedule(event_handler, str(project_root), recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Đang dừng...")
        observer.stop()
    
    observer.join()
    print("✅ Đã dừng theo dõi")


if __name__ == "__main__":
    import sys
    
    delay = 30  # Mặc định đợi 30 giây
    if len(sys.argv) > 1:
        try:
            delay = int(sys.argv[1])
        except ValueError:
            print("⚠️  Delay không hợp lệ, sử dụng mặc định: 30 giây")
    
    watch_and_push(delay=delay)

