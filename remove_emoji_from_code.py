#!/usr/bin/env python3
"""
Script to remove/replace emoji from Python code files
Keeps emoji in markdown files (they're fine there)
"""

import os
import re
import sys
import io
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Emoji mapping to text equivalents
EMOJI_MAP = {
    '✅': '[OK]',
    '❌': '[ERROR]',
    '⚠️': '[WARNING]',
    '🚀': '[START]',
    '📊': '[INFO]',
    '💾': '[SAVE]',
    '🔧': '[CONFIG]',
    '⏳': '[WAIT]',
    '🎯': '[TARGET]',
    '💡': '[TIP]',
    '📈': '[UP]',
    '📉': '[DOWN]',
    '🔄': '[RELOAD]',
    '✨': '[NEW]',
    '🐛': '[BUG]',
    '📝': '[NOTE]',
    '🧪': '[TEST]',
    '🌟': '[STAR]',
    '⭐': '[STAR]',
    '💪': '[STRONG]',
    '🎉': '[SUCCESS]',
    '🔥': '[HOT]',
    '👍': '[GOOD]',
    '📁': '[FOLDER]',
    '📄': '[FILE]',
    '🔍': '[SEARCH]',
    '⚡': '[FAST]',
    '🛠️': '[TOOL]',
    '📦': '[PACKAGE]',
    '🎨': '[STYLE]',
    '♻️': '[RECYCLE]',
    '🔒': '[LOCK]',
    '🔓': '[UNLOCK]',
    '📲': '[DOWNLOAD]',
    '📤': '[UPLOAD]',
    '💻': '[CODE]',
    '🖥️': '[COMPUTER]',
    '🌐': '[GLOBAL]',
    '🔗': '[LINK]',
    '📮': '[POST]',
    '✉️': '[EMAIL]',
    '🗂️': '[ORGANIZE]',
    '📋': '[LIST]',
    '📌': '[PIN]',
    '📍': '[LOCATION]',
    '🏁': '[FINISH]',
    '🎬': '[ACTION]',
    '⏰': '[ALARM]',
    '⏱️': '[TIMER]',
    '🔔': '[NOTIFY]',
    '🔕': '[SILENT]',
    '🔐': '[SECURE]',
    '🔑': '[KEY]',
    '🛡️': '[SHIELD]',
    '⚙️': '[SETTINGS]',
    '🔨': '[BUILD]',
    '🧹': '[CLEAN]',
    '🗑️': '[DELETE]',
    '📥': '[INBOX]',
    '📤': '[OUTBOX]',
    '🚧': '[CONSTRUCTION]',
    '🚨': '[ALERT]',
    '⛔': '[STOP]',
    '🆕': '[NEW]',
    '🆗': '[OK]',
    '🆘': '[SOS]',
    '❗': '[!]',
    '❓': '[?]',
    '💬': '[COMMENT]',
    '💭': '[THOUGHT]',
    '🔀': '[SHUFFLE]',
    '🔁': '[REPEAT]',
    '🔂': '[REPEAT_ONE]',
    '▶️': '[PLAY]',
    '⏸️': '[PAUSE]',
    '⏹️': '[STOP]',
    '⏏️': '[EJECT]',
    '🔼': '[UP]',
    '🔽': '[DOWN]',
}

# Compile regex pattern for all emoji
EMOJI_PATTERN = re.compile('|'.join(re.escape(k) for k in EMOJI_MAP.keys()))


class EmojiRemover:
    """Remove emoji from Python files"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.stats = {
            'files_scanned': 0,
            'files_modified': 0,
            'emoji_replaced': 0,
            'backups_created': 0,
        }
        self.changes: List[Tuple[str, int, str, str]] = []  # (file, line, old, new)
    
    def replace_emoji(self, text: str) -> str:
        """Replace emoji with text equivalents"""
        return EMOJI_PATTERN.sub(lambda m: EMOJI_MAP[m.group(0)], text)
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single file"""
        self.stats['files_scanned'] += 1
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Check if has emoji
            if not EMOJI_PATTERN.search(original_content):
                return False
            
            # Replace emoji line by line to track changes
            lines = original_content.split('\n')
            new_lines = []
            file_has_changes = False
            
            for line_num, line in enumerate(lines, 1):
                if EMOJI_PATTERN.search(line):
                    new_line = self.replace_emoji(line)
                    if new_line != line:
                        file_has_changes = True
                        emoji_count = len(EMOJI_PATTERN.findall(line))
                        self.stats['emoji_replaced'] += emoji_count
                        self.changes.append((str(file_path), line_num, line.strip(), new_line.strip()))
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            
            if not file_has_changes:
                return False
            
            new_content = '\n'.join(new_lines)
            
            if not self.dry_run:
                # Create backup
                backup_dir = Path('emoji_backup') / datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_dir.mkdir(parents=True, exist_ok=True)
                
                # Handle absolute and relative paths
                try:
                    rel_path = file_path.relative_to(Path.cwd())
                except ValueError:
                    # If file is not relative to cwd, use its name with parent dirs
                    rel_path = Path(*file_path.parts[-3:]) if len(file_path.parts) > 3 else file_path.name
                
                backup_path = backup_dir / rel_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
                self.stats['backups_created'] += 1
                
                # Write modified content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            
            self.stats['files_modified'] += 1
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")
            return False
    
    def process_directory(self, directory: Path, exclude_patterns: List[str] = None):
        """Process all Python files in directory"""
        if exclude_patterns is None:
            exclude_patterns = ['venv', 'node_modules', '__pycache__', '.git', 'env']
        
        print(f"\n{'[DRY RUN] ' if self.dry_run else ''}Scanning directory: {directory}")
        print("-" * 60)
        
        for py_file in directory.rglob('*.py'):
            # Skip excluded directories
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue
            
            self.process_file(py_file)
    
    def print_report(self):
        """Print detailed report"""
        print("\n" + "=" * 60)
        print(f"{'DRY RUN ' if self.dry_run else ''}REPORT")
        print("=" * 60)
        
        print(f"\nStatistics:")
        print(f"  Files scanned:    {self.stats['files_scanned']}")
        print(f"  Files modified:   {self.stats['files_modified']}")
        print(f"  Emoji replaced:   {self.stats['emoji_replaced']}")
        print(f"  Backups created:  {self.stats['backups_created']}")
        
        if self.changes:
            print(f"\nChanges Preview (first 20):")
            print("-" * 60)
            for file, line_num, old, new in self.changes[:20]:
                print(f"\n{file}:{line_num}")
                print(f"  - {old[:80]}")
                print(f"  + {new[:80]}")
            
            if len(self.changes) > 20:
                print(f"\n... and {len(self.changes) - 20} more changes")
        
        if self.dry_run:
            print("\n" + "=" * 60)
            print("[DRY RUN] No files were modified.")
            print("Run with --apply to apply changes.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print(f"[SUCCESS] {self.stats['files_modified']} files modified.")
            print(f"Backups saved in: emoji_backup/")
            print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Remove emoji from Python code files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python remove_emoji_from_code.py                    # Dry run (preview only)
  python remove_emoji_from_code.py --apply            # Apply changes
  python remove_emoji_from_code.py --dir backend_api  # Specific directory
  python remove_emoji_from_code.py --apply --all      # All directories
        """
    )
    
    parser.add_argument('--apply', action='store_true',
                        help='Apply changes (default: dry run)')
    parser.add_argument('--dir', type=str,
                        help='Specific directory to process (default: current)')
    parser.add_argument('--all', action='store_true',
                        help='Process all directories (training_experiments, backend_api, dashboard)')
    
    args = parser.parse_args()
    
    remover = EmojiRemover(dry_run=not args.apply)
    
    if args.all:
        # Process all main directories
        directories = [
            Path('training_experiments'),
            Path('backend_api'),
            Path('dashboard'),
        ]
        for directory in directories:
            if directory.exists():
                remover.process_directory(directory)
    elif args.dir:
        directory = Path(args.dir)
        if not directory.exists():
            print(f"[ERROR] Directory not found: {directory}")
            return
        remover.process_directory(directory)
    else:
        # Process current directory
        remover.process_directory(Path.cwd())
    
    remover.print_report()


if __name__ == '__main__':
    main()
