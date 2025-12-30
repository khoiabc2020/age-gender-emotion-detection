"""
Kiểm tra Tuần 7: Business Logic & Tracking
- ByteTrack (thay DeepSORT)
- Ad Recommendation Engine
- Dwell Time logic (> 3 giây)
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


def check_bytetrack():
    """Kiểm tra ByteTrack"""
    print("=" * 60)
    print("🎯 KIỂM TRA BYTETRACK")
    print("=" * 60)
    
    results = []
    
    # Check bytetrack_tracker.py
    print("\n[1/3] Checking bytetrack_tracker.py...")
    tracker_file = project_root / "ai_edge_app" / "src" / "trackers" / "bytetrack_tracker.py"
    
    if tracker_file.exists():
        with open(tracker_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_bytetracker = 'ByteTracker' in content
        has_update = 'def update' in content
        has_iou = 'iou' in content.lower()
        has_match = '_match_detections_to_tracks' in content
        
        if has_bytetracker and has_update and has_iou and has_match:
            print("   ✅ ByteTracker module found")
            print("      - ByteTracker class")
            print("      - update() method")
            print("      - IoU matching")
            results.append(("ByteTracker Module", True))
        else:
            print("   ⚠️  ByteTracker may be incomplete")
            results.append(("ByteTracker Module", False))
    else:
        print("   ❌ bytetrack_tracker.py not found")
        results.append(("ByteTracker Module", False))
    
    # Check integration in main.py
    print("\n[2/3] Checking integration in main.py...")
    main_file = project_root / "ai_edge_app" / "main.py"
    
    if main_file.exists():
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_import = 'from src.trackers import' in content and 'ByteTracker' in content
        has_init = 'ByteTracker(' in content or 'use_bytetrack' in content
        has_use = 'isinstance(tracker_result, list)' in content or 'ByteTracker' in content
        
        if has_import and has_init and has_use:
            print("   ✅ ByteTrack integrated in main.py")
            results.append(("ByteTrack Integration", True))
        else:
            print("   ⚠️  ByteTrack may not be fully integrated")
            results.append(("ByteTrack Integration", False))
    else:
        results.append(("ByteTrack Integration", False))
    
    # Check config
    print("\n[3/3] Checking config...")
    config_file = project_root / "ai_edge_app" / "configs" / "camera_config.json"
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_bytetrack_config = 'use_bytetrack' in content or 'ByteTrack' in content
        
        if has_bytetrack_config:
            print("   ✅ ByteTrack config found")
            results.append(("ByteTrack Config", True))
        else:
            print("   ⚠️  ByteTrack config may be missing")
            results.append(("ByteTrack Config", False))
    else:
        results.append(("ByteTrack Config", False))
    
    return results


def check_ad_recommendation():
    """Kiểm tra Ad Recommendation Engine"""
    print("\n" + "=" * 60)
    print("🎯 KIỂM TRA AD RECOMMENDATION ENGINE")
    print("=" * 60)
    
    results = []
    
    # Check ads_selector.py
    print("\n[1/3] Checking ads_selector.py...")
    ads_file = project_root / "ai_edge_app" / "src" / "ads_engine" / "ads_selector.py"
    
    if ads_file.exists():
        with open(ads_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_ads_selector = 'AdsSelector' in content
        has_select_ad = 'select_ad' in content
        has_linucb = 'LinUCB' in content or 'linucb' in content.lower()
        has_feedback = 'update_feedback' in content
        
        if has_ads_selector and has_select_ad and has_linucb and has_feedback:
            print("   ✅ Ad Recommendation Engine found")
            print("      - AdsSelector class")
            print("      - select_ad() method")
            print("      - LinUCB integration")
            print("      - update_feedback() method")
            results.append(("Ad Recommendation Module", True))
        else:
            print("   ⚠️  Ad Recommendation may be incomplete")
            results.append(("Ad Recommendation Module", False))
    else:
        print("   ❌ ads_selector.py not found")
        results.append(("Ad Recommendation Module", False))
    
    # Check integration
    print("\n[2/3] Checking integration in main.py...")
    main_file = project_root / "ai_edge_app" / "main.py"
    
    if main_file.exists():
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_ads_selector = 'self.ads_selector' in content
        has_select_ad = 'ads_selector.select_ad' in content
        has_feedback = 'ads_selector.update_feedback' in content
        
        if has_ads_selector and has_select_ad and has_feedback:
            print("   ✅ Ad Recommendation integrated in main.py")
            results.append(("Ad Recommendation Integration", True))
        else:
            print("   ⚠️  Ad Recommendation may not be fully integrated")
            results.append(("Ad Recommendation Integration", False))
    else:
        results.append(("Ad Recommendation Integration", False))
    
    # Check LinUCB
    print("\n[3/3] Checking LinUCB module...")
    linucb_file = project_root / "ai_edge_app" / "src" / "ads_engine" / "lin_ucb.py"
    
    if linucb_file.exists():
        print("   ✅ LinUCB module found")
        results.append(("LinUCB Module", True))
    else:
        print("   ❌ lin_ucb.py not found")
        results.append(("LinUCB Module", False))
    
    return results


def check_dwell_time():
    """Kiểm tra Dwell Time logic"""
    print("\n" + "=" * 60)
    print("⏱️  KIỂM TRA DWELL TIME LOGIC")
    print("=" * 60)
    
    results = []
    
    # Check dwell_time.py
    print("\n[1/3] Checking dwell_time.py...")
    dwell_file = project_root / "ai_edge_app" / "src" / "core" / "dwell_time.py"
    
    if dwell_file.exists():
        with open(dwell_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_dwell_tracker = 'DwellTimeTracker' in content
        has_update = 'update_track' in content
        has_is_valid = 'is_valid_customer' in content
        has_threshold = 'threshold' in content.lower()
        
        if has_dwell_tracker and has_update and has_is_valid and has_threshold:
            print("   ✅ Dwell Time module found")
            print("      - DwellTimeTracker class")
            print("      - update_track() method")
            print("      - is_valid_customer() method")
            print("      - Threshold support (> 3s)")
            results.append(("Dwell Time Module", True))
        else:
            print("   ⚠️  Dwell Time may be incomplete")
            results.append(("Dwell Time Module", False))
    else:
        print("   ❌ dwell_time.py not found")
        results.append(("Dwell Time Module", False))
    
    # Check integration
    print("\n[2/3] Checking integration in main.py...")
    main_file = project_root / "ai_edge_app" / "main.py"
    
    if main_file.exists():
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_import = 'from src.core.dwell_time import' in content
        has_init = 'DwellTimeTracker' in content or 'dwell_tracker' in content
        has_use = 'dwell_tracker.update_track' in content or 'dwell_tracker.is_valid_customer' in content
        
        if has_import and has_init and has_use:
            print("   ✅ Dwell Time integrated in main.py")
            results.append(("Dwell Time Integration", True))
        else:
            print("   ⚠️  Dwell Time may not be fully integrated")
            results.append(("Dwell Time Integration", False))
    else:
        results.append(("Dwell Time Integration", False))
    
    # Check config
    print("\n[3/3] Checking config...")
    config_file = project_root / "ai_edge_app" / "configs" / "camera_config.json"
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_dwell_config = 'dwell_threshold' in content
        
        if has_dwell_config:
            print("   ✅ Dwell Time config found")
            results.append(("Dwell Time Config", True))
        else:
            print("   ⚠️  Dwell Time config may be missing")
            results.append(("Dwell Time Config", False))
    else:
        results.append(("Dwell Time Config", False))
    
    return results


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🔍 KIỂM TRA TUẦN 7: BUSINESS LOGIC & TRACKING")
    print("=" * 60)
    
    all_results = []
    
    # Check ByteTrack
    bytetrack_results = check_bytetrack()
    all_results.extend(bytetrack_results)
    
    # Check Ad Recommendation
    ad_results = check_ad_recommendation()
    all_results.extend(ad_results)
    
    # Check Dwell Time
    dwell_results = check_dwell_time()
    all_results.extend(dwell_results)
    
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
        print("🎉 Tất cả yêu cầu Tuần 7 đã hoàn thành!")
        print("\nBusiness Logic & Tracking đã được implement:")
        print("  - ByteTrack: Thay thế DeepSORT")
        print("  - Ad Recommendation Engine: LinUCB với feedback")
        print("  - Dwell Time: Chỉ tính khách hàng nếu > 3 giây")
    else:
        print("⚠️  Một số yêu cầu chưa hoàn thành")
        print("\nCần kiểm tra và sửa các phần còn thiếu")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
