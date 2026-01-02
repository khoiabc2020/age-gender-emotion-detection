"""
Build script to create Windows .exe installer for Edge AI Application
Uses PyInstaller to create standalone executable
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required = ['pyinstaller', 'pyinstaller-hooks-contrib']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("Dependencies installed successfully!")

def create_spec_file():
    """Create PyInstaller spec file for the application"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('configs', 'configs'),
        ('models', 'models'),
    ],
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'cv2',
        'numpy',
        'onnxruntime',
        'PIL',
        'qrcode',
        'paho.mqtt',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SmartRetailAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
'''
    
    with open('SmartRetailAI.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    print("Created SmartRetailAI.spec file")

def build_exe():
    """Build the executable"""
    print("Building executable...")
    print("This may take several minutes...")
    
    try:
        # Clean previous builds
        if os.path.exists('build'):
            shutil.rmtree('build')
        if os.path.exists('dist'):
            shutil.rmtree('dist')
        
        # Build with PyInstaller
        subprocess.check_call([
            sys.executable, '-m', 'PyInstaller',
            '--clean',
            '--noconfirm',
            'SmartRetailAI.spec'
        ])
        
        print("\n" + "="*60)
        print("Build completed successfully!")
        print("="*60)
        print(f"Executable location: {os.path.abspath('dist/SmartRetailAI.exe')}")
        print("\nYou can now:")
        print("1. Test the executable: dist/SmartRetailAI.exe")
        print("2. Create installer using: python create_installer.py")
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    print("="*60)
    print("Smart Retail AI - Windows Executable Builder")
    print("="*60)
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Check and install dependencies
    check_dependencies()
    
    # Create spec file
    create_spec_file()
    
    # Build executable
    build_exe()
