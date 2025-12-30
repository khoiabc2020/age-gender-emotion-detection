"""
Kiểm tra Tuần 4: Setup UI Framework
- PyQt6 + QFluentWidgets
- Glassmorphism (Acrylic effect)
- Dashboard HUD
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
sys.path.insert(0, str(project_root / "ai_edge_app" / "src"))


def check_pyqt6_qfluent():
    """Kiểm tra PyQt6 và QFluentWidgets"""
    print("=" * 60)
    print("🎨 KIỂM TRA PYQT6 + QFLUENTWIDGETS")
    print("=" * 60)
    
    results = []
    
    # Check requirements.txt
    print("\n[1/3] Checking requirements.txt...")
    req_file = project_root / "ai_edge_app" / "requirements.txt"
    
    if req_file.exists():
        with open(req_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_pyqt6 = 'PyQt6' in content
        has_qfluent = 'qfluentwidgets' in content.lower()
        
        if has_pyqt6 and has_qfluent:
            print("   ✅ PyQt6 and QFluentWidgets in requirements")
            results.append(("Requirements", True))
        else:
            print("   ⚠️  PyQt6/QFluentWidgets may be missing")
            results.append(("Requirements", False))
    else:
        results.append(("Requirements", False))
    
    # Check main_window.py
    print("\n[2/3] Checking main_window.py...")
    main_window_file = project_root / "ai_edge_app" / "src" / "ui" / "main_window.py"
    
    if main_window_file.exists():
        with open(main_window_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_pyqt6_import = 'from PyQt6' in content
        has_qfluent_import = 'from qfluentwidgets' in content
        has_fluent_window = 'FluentWindow' in content
        has_main_window = 'class MainWindow' in content
        
        if has_pyqt6_import and has_qfluent_import and has_fluent_window and has_main_window:
            print("   ✅ MainWindow with PyQt6 + QFluentWidgets found")
            results.append(("MainWindow Implementation", True))
        else:
            print("   ⚠️  MainWindow may be incomplete")
            results.append(("MainWindow Implementation", False))
    else:
        print("   ❌ main_window.py not found")
        results.append(("MainWindow Implementation", False))
    
    # Check if can import
    print("\n[3/3] Testing import...")
    try:
        from PyQt6.QtWidgets import QApplication
        from qfluentwidgets import FluentWindow, setTheme, Theme
        print("   ✅ PyQt6 and QFluentWidgets can be imported")
        results.append(("Import Test", True))
    except ImportError as e:
        print(f"   ❌ Cannot import: {e}")
        print("      Install: pip install PyQt6 qfluentwidgets")
        results.append(("Import Test", False))
    
    return results


def check_glassmorphism():
    """Kiểm tra Glassmorphism effect"""
    print("\n" + "=" * 60)
    print("✨ KIỂM TRA GLASSMORPHISM (ACRYLIC EFFECT)")
    print("=" * 60)
    
    results = []
    
    # Check glassmorphism.py
    print("\n[1/3] Checking glassmorphism.py...")
    glass_file = project_root / "ai_edge_app" / "src" / "ui" / "glassmorphism.py"
    
    if glass_file.exists():
        with open(glass_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_stylesheet = 'GLASSMORPHISM_STYLESHEET' in content
        has_backdrop = 'backdrop-filter' in content or 'backdropFilter' in content
        has_apply = 'apply_glassmorphism' in content
        has_rgba = 'rgba' in content
        
        if has_stylesheet and has_backdrop and has_apply and has_rgba:
            print("   ✅ Glassmorphism module found")
            print("      - GLASSMORPHISM_STYLESHEET")
            print("      - apply_glassmorphism() function")
            print("      - Backdrop filter support")
            results.append(("Glassmorphism Module", True))
        else:
            print("   ⚠️  Glassmorphism may be incomplete")
            results.append(("Glassmorphism Module", False))
    else:
        print("   ❌ glassmorphism.py not found")
        results.append(("Glassmorphism Module", False))
    
    # Check usage in main_window.py
    print("\n[2/3] Checking usage in main_window.py...")
    main_window_file = project_root / "ai_edge_app" / "src" / "ui" / "main_window.py"
    
    if main_window_file.exists():
        with open(main_window_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_import = 'from .glassmorphism import' in content
        has_apply = 'apply_glassmorphism' in content
        has_stylesheet = 'GLASSMORPHISM_STYLESHEET' in content
        
        if has_import and (has_apply or has_stylesheet):
            print("   ✅ Glassmorphism integrated in MainWindow")
            results.append(("Glassmorphism Integration", True))
        else:
            print("   ⚠️  Glassmorphism may not be applied")
            results.append(("Glassmorphism Integration", False))
    else:
        results.append(("Glassmorphism Integration", False))
    
    # Check if can import
    print("\n[3/3] Testing import...")
    try:
        from ui.glassmorphism import apply_glassmorphism, GLASSMORPHISM_STYLESHEET
        print("   ✅ Glassmorphism can be imported")
        results.append(("Glassmorphism Import", True))
    except Exception as e:
        print(f"   ❌ Cannot import: {e}")
        results.append(("Glassmorphism Import", False))
    
    return results


def check_dashboard_hud():
    """Kiểm tra Dashboard HUD"""
    print("\n" + "=" * 60)
    print("📊 KIỂM TRA DASHBOARD HUD")
    print("=" * 60)
    
    results = []
    
    # Check hud_overlay.py
    print("\n[1/3] Checking hud_overlay.py...")
    hud_file = project_root / "ai_edge_app" / "src" / "ui" / "hud_overlay.py"
    
    if hud_file.exists():
        with open(hud_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_hud = 'HUDOverlay' in content or 'class HUD' in content
        has_fps = 'update_fps' in content
        has_stats = 'update_stats' in content
        has_paint = 'paintEvent' in content
        
        if has_hud and has_fps and has_stats and has_paint:
            print("   ✅ HUD Overlay module found")
            print("      - HUDOverlay class")
            print("      - update_fps() method")
            print("      - update_stats() method")
            print("      - Custom paintEvent")
            results.append(("HUD Module", True))
        else:
            print("   ⚠️  HUD may be incomplete")
            results.append(("HUD Module", False))
    else:
        print("   ❌ hud_overlay.py not found")
        results.append(("HUD Module", False))
    
    # Check usage in main_window.py
    print("\n[2/3] Checking usage in main_window.py...")
    main_window_file = project_root / "ai_edge_app" / "src" / "ui" / "main_window.py"
    
    if main_window_file.exists():
        with open(main_window_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_import = 'from .hud_overlay import' in content or 'HUDOverlay' in content
        has_init = 'self.hud_overlay' in content
        has_update = 'hud_overlay.update' in content
        
        if has_import and has_init and has_update:
            print("   ✅ HUD integrated in MainWindow")
            results.append(("HUD Integration", True))
        else:
            print("   ⚠️  HUD may not be fully integrated")
            results.append(("HUD Integration", False))
    else:
        results.append(("HUD Integration", False))
    
    # Check StatsCardWidget
    print("\n[3/3] Checking StatsCardWidget...")
    main_window_file = project_root / "ai_edge_app" / "src" / "ui" / "main_window.py"
    
    if main_window_file.exists():
        with open(main_window_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_stats_card = 'StatsCardWidget' in content
        has_fps_card = 'fps_card' in content
        has_customer_card = 'customer_card' in content
        
        if has_stats_card and has_fps_card and has_customer_card:
            print("   ✅ Stats cards implemented")
            results.append(("Stats Cards", True))
        else:
            print("   ⚠️  Stats cards may be missing")
            results.append(("Stats Cards", False))
    else:
        results.append(("Stats Cards", False))
    
    return results


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🔍 KIỂM TRA TUẦN 4: SETUP UI FRAMEWORK")
    print("=" * 60)
    
    all_results = []
    
    # Check PyQt6 + QFluentWidgets
    pyqt_results = check_pyqt6_qfluent()
    all_results.extend(pyqt_results)
    
    # Check Glassmorphism
    glass_results = check_glassmorphism()
    all_results.extend(glass_results)
    
    # Check Dashboard HUD
    hud_results = check_dashboard_hud()
    all_results.extend(hud_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT")
    print("=" * 60)
    
    passed = sum(1 for _, result in all_results if result)
    total = len(all_results)
    
    print(f"\nKết quả: {passed}/{total} checks passed\n")
    
    for name, result in all_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:50s} {status}")
    
    print("\n" + "=" * 60)
    
    if passed == total:
        print("🎉 Tất cả yêu cầu Tuần 4 đã hoàn thành!")
        print("\nUI Framework đã được setup:")
        print("  - PyQt6 + QFluentWidgets")
        print("  - Glassmorphism (Acrylic effect)")
        print("  - Dashboard HUD overlay")
        print("  - Stats cards")
    else:
        print("⚠️  Một số yêu cầu chưa hoàn thành")
        print("\nCần kiểm tra và sửa các phần còn thiếu")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)




