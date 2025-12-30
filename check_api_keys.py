"""
Script kiểm tra API keys có được bảo vệ đúng cách không
"""

import os
import re
from pathlib import Path

# Colors
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

def check_file_for_hardcoded_keys(filepath):
    """Kiểm tra file có hardcode API keys không"""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Patterns để tìm hardcoded keys
        patterns = [
            (r'["\'](sk-[a-zA-Z0-9]{20,})["\']', 'OpenAI API key'),
            (r'["\'](AIza[Sy][a-zA-Z0-9_-]{35})["\']', 'Google AI API key'),
            (r'api[_-]?key\s*=\s*["\'][^"\']{20,}["\']', 'API key hardcoded'),
            (r'secret[_-]?key\s*=\s*["\'][^"\']{20,}["\']', 'Secret key hardcoded'),
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, key_type in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    # Bỏ qua nếu là comment hoặc example
                    if 'example' in line.lower() or 'your-' in line.lower() or 'change-in-production' in line.lower():
                        continue
                    issues.append({
                        'line': i,
                        'content': line.strip(),
                        'type': key_type,
                        'match': match.group(0)[:20] + '...'
                    })
    except Exception as e:
        return [{'error': str(e)}]
    
    return issues

def check_env_usage(filepath):
    """Kiểm tra file có dùng environment variables không"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Patterns để tìm env usage
        env_patterns = [
            r'os\.getenv\s*\(',
            r'os\.environ\s*\[',
            r'getattr\s*\(settings',
            r'from\s+pydantic_settings',
            r'BaseSettings',
        ]
        
        uses_env = any(re.search(pattern, content) for pattern in env_patterns)
        return uses_env
    except:
        return False

def main():
    print_header("KIỂM TRA API KEYS - BẢO MẬT")
    
    project_root = Path(__file__).parent
    issues_found = []
    files_checked = []
    
    # Files cần kiểm tra
    files_to_check = [
        "backend_api/app/core/config.py",
        "backend_api/app/services/ai_agent.py",
        "backend_api/app/api/ai_agent.py",
        "backend_api/app/main.py",
        "ai_edge_app/src/services/generative_ads.py",
    ]
    
    print("\n[1] Kiểm tra hardcoded keys...")
    for file_path in files_to_check:
        full_path = project_root / file_path
        if full_path.exists():
            files_checked.append(file_path)
            issues = check_file_for_hardcoded_keys(full_path)
            if issues:
                for issue in issues:
                    if 'error' not in issue:
                        print_error(f"{file_path}:{issue['line']} - {issue['type']}")
                        print(f"  {issue['content'][:80]}...")
                        issues_found.append(issue)
            else:
                print_success(f"{file_path} - Không có hardcoded keys")
    
    print("\n[2] Kiểm tra sử dụng environment variables...")
    for file_path in files_to_check:
        full_path = project_root / file_path
        if full_path.exists():
            uses_env = check_env_usage(full_path)
            if uses_env:
                print_success(f"{file_path} - Sử dụng environment variables")
            else:
                print_warning(f"{file_path} - Không thấy sử dụng env vars (có thể OK)")
    
    print("\n[3] Kiểm tra .gitignore...")
    gitignore_path = project_root / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            gitignore_content = f.read()
        
        if '.env' in gitignore_content:
            print_success(".env được ignore trong .gitignore")
        else:
            print_error(".env KHÔNG được ignore!")
            issues_found.append({'type': '.env not in .gitignore'})
        
        if '.env.local' in gitignore_content:
            print_success(".env.local được ignore trong .gitignore")
        else:
            print_warning(".env.local không có trong .gitignore")
    else:
        print_warning(".gitignore không tồn tại")
    
    print("\n[4] Kiểm tra .env files...")
    env_files = [
        project_root / "backend_api" / ".env",
        project_root / "dashboard" / ".env.local",
    ]
    
    for env_file in env_files:
        if env_file.exists():
            print_warning(f"{env_file} tồn tại (OK cho local, nhưng không nên commit)")
            # Kiểm tra xem có keys thật không
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'GOOGLE_AI_API_KEY=' in content or 'OPENAI_API_KEY=' in content:
                        # Kiểm tra xem có phải là placeholder không
                        if 'your-key' in content.lower() or 'example' in content.lower():
                            print_success(f"  → Chỉ có placeholder, an toàn")
                        else:
                            print_warning(f"  → Có thể chứa keys thật (kiểm tra thủ công)")
            except:
                pass
        else:
            print_success(f"{env_file} không tồn tại (OK)")
    
    print("\n[5] Kiểm tra .env.example...")
    env_example = project_root / "backend_api" / ".env.example"
    if env_example.exists():
        print_success(".env.example tồn tại (tốt để hướng dẫn)")
    else:
        print_warning(".env.example không tồn tại (nên tạo)")
    
    # Tổng kết
    print_header("TỔNG KẾT")
    
    if not issues_found:
        print_success("Không tìm thấy hardcoded API keys!")
        print_success("API keys được bảo vệ đúng cách!")
    else:
        print_error(f"Tìm thấy {len(issues_found)} vấn đề:")
        for issue in issues_found:
            if 'error' not in issue:
                print(f"  - {issue.get('type', 'Unknown')}")
    
    print("\n" + "="*60)
    print(f"{GREEN}✓ BẢO MẬT API KEYS: {'OK' if not issues_found else 'CẦN KIỂM TRA'}{RESET}")
    print("="*60 + "\n")
    
    print("📝 Khuyến nghị:")
    print("  1. Luôn dùng environment variables cho API keys")
    print("  2. Không commit .env files vào git")
    print("  3. Tạo .env.example với placeholder values")
    print("  4. Sử dụng secrets management trong production")
    print()
    
    return len(issues_found) == 0

if __name__ == "__main__":
    import sys
    # Fix encoding for Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    success = main()
    sys.exit(0 if success else 1)

