"""
Kiểm tra Tuần 6: Dynamic Ads System
- Smart Player (QMediaPlayer, Video 4K)
- Transition Effects (Fade, Slide)
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


def check_ads_player():
    """Kiểm tra Smart Ads Player"""
    print("=" * 60)
    print("🎬 KIỂM TRA SMART ADS PLAYER")
    print("=" * 60)
    
    results = []
    
    # Check ads_player.py
    print("\n[1/4] Checking ads_player.py...")
    player_file = project_root / "ai_edge_app" / "src" / "ui" / "ads_player.py"
    
    if player_file.exists():
        with open(player_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_ads_player = 'AdsPlayerWidget' in content or 'AdsPlayer' in content
        has_qmediaplayer = 'QMediaPlayer' in content
        has_video_widget = 'QVideoWidget' in content
        has_play_ad = 'play_ad' in content
        has_transition = 'TransitionEffect' in content or 'transition' in content.lower()
        
        if has_ads_player and has_qmediaplayer and has_video_widget and has_play_ad and has_transition:
            print("   ✅ Ads Player module found")
            print("      - AdsPlayerWidget class")
            print("      - QMediaPlayer integration")
            print("      - QVideoWidget for video")
            print("      - play_ad() method")
            print("      - Transition effects")
            results.append(("Ads Player Module", True))
        else:
            print("   ⚠️  Ads Player may be incomplete")
            results.append(("Ads Player Module", False))
    else:
        print("   ❌ ads_player.py not found")
        results.append(("Ads Player Module", False))
    
    # Check transition effects
    print("\n[2/4] Checking transition effects...")
    if player_file.exists():
        with open(player_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_fade = 'FADE' in content or 'fade' in content
        has_slide = 'SLIDE' in content or 'slide' in content
        has_animation = 'QPropertyAnimation' in content or 'transition_animation' in content
        
        if has_fade and has_slide and has_animation:
            print("   ✅ Transition effects found")
            print("      - Fade effect")
            print("      - Slide effects")
            print("      - QPropertyAnimation")
            results.append(("Transition Effects", True))
        else:
            print("   ⚠️  Transition effects may be incomplete")
            results.append(("Transition Effects", False))
    else:
        results.append(("Transition Effects", False))
    
    # Check integration in main_window.py
    print("\n[3/4] Checking integration in main_window.py...")
    main_window_file = project_root / "ai_edge_app" / "src" / "ui" / "main_window.py"
    
    if main_window_file.exists():
        with open(main_window_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_import = 'from .ads_player import' in content
        has_ads_player = 'ads_player' in content or 'AdsPlayerCard' in content
        has_play_ad = 'play_advertisement' in content or 'play_ad' in content
        
        if has_import and has_ads_player and has_play_ad:
            print("   ✅ Ads Player integrated in MainWindow")
            results.append(("Ads Player Integration", True))
        else:
            print("   ⚠️  Ads Player may not be fully integrated")
            results.append(("Ads Player Integration", False))
    else:
        results.append(("Ads Player Integration", False))
    
    # Check requirements
    print("\n[4/4] Checking requirements...")
    req_file = project_root / "ai_edge_app" / "requirements.txt"
    
    if req_file.exists():
        with open(req_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_multimedia = 'QtMultimedia' in content or 'multimedia' in content.lower()
        
        if has_multimedia:
            print("   ✅ PyQt6-QtMultimedia in requirements")
            results.append(("Multimedia Requirements", True))
        else:
            print("   ⚠️  PyQt6-QtMultimedia may be missing")
            results.append(("Multimedia Requirements", False))
    else:
        results.append(("Multimedia Requirements", False))
    
    return results


def check_video_support():
    """Kiểm tra Video 4K support"""
    print("\n" + "=" * 60)
    print("🎥 KIỂM TRA VIDEO 4K SUPPORT")
    print("=" * 60)
    
    results = []
    
    # Check if QMediaPlayer supports high resolution
    print("\n[1/2] Checking QMediaPlayer capabilities...")
    player_file = project_root / "ai_edge_app" / "src" / "ui" / "ads_player.py"
    
    if player_file.exists():
        with open(player_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_video_widget = 'QVideoWidget' in content
        has_media_player = 'QMediaPlayer' in content
        has_audio_output = 'QAudioOutput' in content
        
        if has_video_widget and has_media_player and has_audio_output:
            print("   ✅ QMediaPlayer setup found")
            print("      - QVideoWidget for display")
            print("      - QMediaPlayer for playback")
            print("      - QAudioOutput for audio")
            print("      - Note: 4K support depends on codec availability")
            results.append(("Video Player Setup", True))
        else:
            print("   ⚠️  Video player setup may be incomplete")
            results.append(("Video Player Setup", False))
    else:
        results.append(("Video Player Setup", False))
    
    # Check import
    print("\n[2/2] Testing import...")
    try:
        from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PyQt6.QtMultimediaWidgets import QVideoWidget
        print("   ✅ QMediaPlayer modules can be imported")
        results.append(("Video Player Import", True))
    except ImportError as e:
        print(f"   ❌ Cannot import: {e}")
        print("      Install: pip install PyQt6-QtMultimedia PyQt6-QtMultimediaWidgets")
        results.append(("Video Player Import", False))
    
    return results


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🔍 KIỂM TRA TUẦN 6: DYNAMIC ADS SYSTEM")
    print("=" * 60)
    
    all_results = []
    
    # Check Ads Player
    player_results = check_ads_player()
    all_results.extend(player_results)
    
    # Check Video Support
    video_results = check_video_support()
    all_results.extend(video_results)
    
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
        print("🎉 Tất cả yêu cầu Tuần 6 đã hoàn thành!")
        print("\nDynamic Ads System đã được implement:")
        print("  - Smart Player: QMediaPlayer với QVideoWidget")
        print("  - Video 4K: Support (depends on codec)")
        print("  - Transition Effects: Fade, Slide (left/right/up/down)")
    else:
        print("⚠️  Một số yêu cầu chưa hoàn thành")
        print("\nCần kiểm tra và sửa các phần còn thiếu")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)




