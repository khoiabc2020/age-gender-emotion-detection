#!/usr/bin/env python3
"""
Automatic Code Cleanup Script
Removes AI-generated markers, Vietnamese comments, and improves professionalism
"""

import re
import os
from pathlib import Path
from typing import List, Tuple

class CodeCleaner:
    """Clean code to make it more professional"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.stats = {
            'files_processed': 0,
            'lines_cleaned': 0,
            'comments_removed': 0
        }
    
    def clean_python_file(self, filepath: Path) -> Tuple[str, int]:
        """Clean a Python file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_lines = content.count('\n')
        
        # Remove "Tuần X" markers
        content = re.sub(r'Tuần \d+[:\-]?\s*', '', content)
        content = re.sub(r'# Tuần \d+.*\n', '', content)
        
        # Remove excessive Vietnamese comments
        content = re.sub(r'#.*\(Optimized:.*\)', '', content)
        content = re.sub(r'# TODO: .*tiếng Việt.*\n', '', content)
        
        # Remove emoji from comments
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE)
        content = emoji_pattern.sub('', content)
        
        # Clean up excessive newlines
        content = re.sub(r'\n\n\n+', '\n\n', content)
        
        cleaned_lines = content.count('\n')
        lines_removed = original_lines - cleaned_lines
        
        return content, lines_removed
    
    def clean_javascript_file(self, filepath: Path) -> Tuple[str, int]:
        """Clean a JavaScript/JSX file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_lines = content.count('\n')
        
        # Remove Vietnamese comments
        content = re.sub(r'//.*tiếng Việt.*\n', '', content)
        content = re.sub(r'/\*.*Tuần \d+.*\*/', '', content, flags=re.DOTALL)
        
        # Remove emoji
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE)
        content = emoji_pattern.sub('', content)
        
        # Clean up excessive newlines
        content = re.sub(r'\n\n\n+', '\n\n', content)
        
        cleaned_lines = content.count('\n')
        lines_removed = original_lines - cleaned_lines
        
        return content, lines_removed
    
    def process_directory(self, directory: Path, extensions: List[str]):
        """Process all files in directory"""
        for ext in extensions:
            for filepath in directory.rglob(f'*{ext}'):
                # Skip node_modules, venv, __pycache__, etc.
                if any(skip in str(filepath) for skip in ['node_modules', 'venv', '__pycache__', '.git', 'checkpoints', 'data']):
                    continue
                
                try:
                    if ext in ['.py']:
                        cleaned_content, lines_removed = self.clean_python_file(filepath)
                    elif ext in ['.js', '.jsx']:
                        cleaned_content, lines_removed = self.clean_javascript_file(filepath)
                    else:
                        continue
                    
                    # Write back
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    
                    self.stats['files_processed'] += 1
                    self.stats['lines_cleaned'] += lines_removed
                    
                    if lines_removed > 0:
                        print(f"✓ Cleaned: {filepath} ({lines_removed} lines removed)")
                
                except Exception as e:
                    print(f"✗ Error processing {filepath}: {e}")
    
    def run(self):
        """Run the cleanup"""
        print("=" * 60)
        print("CODE CLEANUP - Making code professional")
        print("=" * 60)
        
        # Process Python files
        print("\n[1/2] Cleaning Python files...")
        for module in ['ai_edge_app', 'backend_api', 'training_experiments']:
            module_path = self.project_root / module
            if module_path.exists():
                self.process_directory(module_path, ['.py'])
        
        # Process JavaScript files
        print("\n[2/2] Cleaning JavaScript files...")
        dashboard_path = self.project_root / 'dashboard'
        if dashboard_path.exists():
            self.process_directory(dashboard_path, ['.js', '.jsx'])
        
        # Print stats
        print("\n" + "=" * 60)
        print("CLEANUP COMPLETE")
        print("=" * 60)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Lines cleaned: {self.stats['lines_cleaned']}")
        print("=" * 60)


if __name__ == '__main__':
    cleaner = CodeCleaner()
    cleaner.run()
