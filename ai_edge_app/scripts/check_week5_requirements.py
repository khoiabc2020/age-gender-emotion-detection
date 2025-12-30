"""
Kiểm tra Tuần 5: Real-time Visualization
- Smart Overlay (Bounding Box bo tròn, màu theo cảm xúc)
- Live Charts (PyQtGraph)
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


def check_smart_overlay():
    """Kiểm tra Smart Overlay"""
    print("=" * 60)
    print("🎨 KIỂM TRA SMART OVERLAY")
    print("=" * 60)
    
    results = []
    
    # Check smart_overlay.py
    print("\n[1/3] Checking smart_overlay.py...")
    overlay_file = project_root / "ai_edge_app" / "src" / "ui" / "smart_overlay.py"
    
    if overlay_file.exists():
        with open(overlay_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_smart_overlay = 'SmartOverlay' in content
        has_rounded = 'rounded' in content.lower() or 'RoundedRect' in content
        has_emotion_colors = 'EMOTION_COLORS' in content
        has_draw = 'draw_track_overlay' in content or 'draw_multiple_tracks' in content
        
        if has_smart_overlay and has_rounded and has_emotion_colors and has_draw:
            print("   ✅ Smart Overlay module found")
            print("      - SmartOverlay class")
            print("      - Rounded rectangles")
            print("      - Emotion colors")
            print("      - Draw methods")
            results.append(("Smart Overlay Module", True))
        else:
            print("   ⚠️  Smart Overlay may be incomplete")
            results.append(("Smart Overlay Module", False))
    else:
        print("   ❌ smart_overlay.py not found")
        results.append(("Smart Overlay Module", False))
    
    # Check integration in main_window.py
    print("\n[2/3] Checking integration in main_window.py...")
    main_window_file = project_root / "ai_edge_app" / "src" / "ui" / "main_window.py"
    
    if main_window_file.exists():
        with open(main_window_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_import = 'from .smart_overlay import' in content
        has_init = 'overlay_renderer' in content or 'SmartOverlay' in content
        has_use = '_draw_smart_overlay' in content or 'draw_multiple_tracks' in content
        
        if has_import and has_init and has_use:
            print("   ✅ Smart Overlay integrated in MainWindow")
            results.append(("Smart Overlay Integration", True))
        else:
            print("   ⚠️  Smart Overlay may not be fully integrated")
            results.append(("Smart Overlay Integration", False))
    else:
        results.append(("Smart Overlay Integration", False))
    
    # Check if can import
    print("\n[3/3] Testing import...")
    try:
        from ui.smart_overlay import SmartOverlay
        overlay = SmartOverlay()
        print("   ✅ SmartOverlay can be imported")
        results.append(("Smart Overlay Import", True))
    except Exception as e:
        print(f"   ❌ Cannot import: {e}")
        results.append(("Smart Overlay Import", False))
    
    return results


def check_live_charts():
    """Kiểm tra Live Charts (PyQtGraph)"""
    print("\n" + "=" * 60)
    print("📊 KIỂM TRA LIVE CHARTS (PYQTGRAPH)")
    print("=" * 60)
    
    results = []
    
    # Check live_charts.py
    print("\n[1/4] Checking live_charts.py...")
    charts_file = project_root / "ai_edge_app" / "src" / "ui" / "live_charts.py"
    
    if charts_file.exists():
        with open(charts_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_emotion_chart = 'EmotionDistributionChart' in content
        has_customer_chart = 'CustomerFlowChart' in content
        has_fps_chart = 'FPSChart' in content
        has_pyqtgraph = 'pyqtgraph' in content or 'pg.' in content
        
        if has_emotion_chart and has_customer_chart and has_fps_chart and has_pyqtgraph:
            print("   ✅ Live Charts module found")
            print("      - EmotionDistributionChart")
            print("      - CustomerFlowChart")
            print("      - FPSChart")
            print("      - PyQtGraph integration")
            results.append(("Live Charts Module", True))
        else:
            print("   ⚠️  Live Charts may be incomplete")
            results.append(("Live Charts Module", False))
    else:
        print("   ❌ live_charts.py not found")
        results.append(("Live Charts Module", False))
    
    # Check integration in main_window.py
    print("\n[2/4] Checking integration in main_window.py...")
    main_window_file = project_root / "ai_edge_app" / "src" / "ui" / "main_window.py"
    
    if main_window_file.exists():
        with open(main_window_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_import = 'from .live_charts import' in content
        has_emotion_chart = 'emotion_chart' in content
        has_customer_chart = 'customer_flow_chart' in content
        has_fps_chart = 'fps_chart' in content
        
        if has_import and has_emotion_chart and has_customer_chart and has_fps_chart:
            print("   ✅ Live Charts integrated in MainWindow")
            results.append(("Live Charts Integration", True))
        else:
            print("   ⚠️  Live Charts may not be fully integrated")
            results.append(("Live Charts Integration", False))
    else:
        results.append(("Live Charts Integration", False))
    
    # Check PyQtGraph in requirements
    print("\n[3/4] Checking PyQtGraph in requirements...")
    req_file = project_root / "ai_edge_app" / "requirements.txt"
    
    if req_file.exists():
        with open(req_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_pyqtgraph = 'PyQtGraph' in content or 'pyqtgraph' in content.lower()
        
        if has_pyqtgraph:
            print("   ✅ PyQtGraph in requirements")
            results.append(("PyQtGraph Requirements", True))
        else:
            print("   ⚠️  PyQtGraph may be missing from requirements")
            results.append(("PyQtGraph Requirements", False))
    else:
        results.append(("PyQtGraph Requirements", False))
    
    # Check if can import
    print("\n[4/4] Testing import...")
    try:
        import pyqtgraph as pg
        from ui.live_charts import EmotionDistributionChart, CustomerFlowChart, FPSChart
        print(f"   ✅ PyQtGraph and charts can be imported (v{pg.__version__})")
        results.append(("Live Charts Import", True))
    except ImportError as e:
        print(f"   ❌ Cannot import: {e}")
        print("      Install: pip install PyQtGraph")
        results.append(("Live Charts Import", False))
    
    return results


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🔍 KIỂM TRA TUẦN 5: REAL-TIME VISUALIZATION")
    print("=" * 60)
    
    all_results = []
    
    # Check Smart Overlay
    overlay_results = check_smart_overlay()
    all_results.extend(overlay_results)
    
    # Check Live Charts
    charts_results = check_live_charts()
    all_results.extend(charts_results)
    
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
        print("🎉 Tất cả yêu cầu Tuần 5 đã hoàn thành!")
        print("\nReal-time Visualization đã được implement:")
        print("  - Smart Overlay: Rounded boxes với emotion colors")
        print("  - Live Charts: Emotion, Customer Flow, FPS charts")
    else:
        print("⚠️  Một số yêu cầu chưa hoàn thành")
        print("\nCần kiểm tra và sửa các phần còn thiếu")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)




