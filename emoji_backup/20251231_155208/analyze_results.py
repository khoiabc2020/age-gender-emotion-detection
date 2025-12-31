"""
Phân tích và đánh giá kết quả training 10 lần
Tự động tạo báo cáo chi tiết và đánh giá
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# Fix encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


def load_results(results_dir: Path) -> Optional[Dict]:
    """Load kết quả từ summary.json"""
    summary_file = results_dir / "summary.json"
    if not summary_file.exists():
        return None
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_results(summary: Dict) -> Dict:
    """Phân tích kết quả và tạo đánh giá"""
    results = summary.get('results', [])
    
    if not results:
        return {
            'status': 'no_results',
            'message': 'Chưa có kết quả training'
        }
    
    # Phân loại kết quả
    successful_runs = [r for r in results if r.get('success', False)]
    failed_runs = [r for r in results if not r.get('success', False)]
    
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'total_runs': len(results),
        'successful_runs': len(successful_runs),
        'failed_runs': len(failed_runs),
        'success_rate': len(successful_runs) / len(results) * 100 if results else 0,
        'best_run': None,
        'config_analysis': {},
        'recommendations': []
    }
    
    # Phân tích từng config
    config_stats = {}
    for result in successful_runs:
        config = result.get('config', {})
        config_key = f"LR={config.get('learning_rate', 0)}, Batch={config.get('batch_size', 0)}, Epochs={config.get('epochs', 0)}"
        
        if config_key not in config_stats:
            config_stats[config_key] = {
                'count': 0,
                'avg_time': 0,
                'runs': []
            }
        
        config_stats[config_key]['count'] += 1
        config_stats[config_key]['avg_time'] += result.get('elapsed_time', 0)
        config_stats[config_key]['runs'].append(result)
    
    # Tính average time
    for key in config_stats:
        if config_stats[key]['count'] > 0:
            config_stats[key]['avg_time'] /= config_stats[key]['count']
    
    analysis['config_analysis'] = config_stats
    
    # Tìm best run
    if successful_runs:
        # Sort by test_accuracy if available, else by elapsed_time
        if any('test_accuracy' in r for r in successful_runs):
            best = max(successful_runs, key=lambda x: x.get('test_accuracy', 0))
        else:
            # Sort by success rate and time
            best = min(successful_runs, key=lambda x: x.get('elapsed_time', float('inf')))
        
        analysis['best_run'] = {
            'run_id': best.get('run_id'),
            'config': best.get('config'),
            'elapsed_time': best.get('elapsed_time'),
            'test_accuracy': best.get('test_accuracy', 'N/A')
        }
    
    # Đưa ra recommendations
    if len(failed_runs) > len(successful_runs):
        analysis['recommendations'].append("⚠️ Nhiều lần training thất bại. Kiểm tra lại dữ liệu và dependencies.")
    
    if successful_runs:
        analysis['recommendations'].append("✅ Training thành công! Có thể sử dụng best model.")
    
    # Phân tích config tốt nhất
    if config_stats:
        best_config_key = min(config_stats.keys(), key=lambda k: config_stats[k]['avg_time'])
        analysis['recommendations'].append(f"💡 Config nhanh nhất: {best_config_key}")
    
    return analysis


def print_analysis(analysis: Dict):
    """In ra phân tích kết quả"""
    print("\n" + "=" * 80)
    print("📊 PHÂN TÍCH KẾT QUẢ TRAINING")
    print("=" * 80)
    
    print(f"\n📈 Tổng quan:")
    print(f"  - Tổng số lần chạy: {analysis['total_runs']}")
    print(f"  - Số lần thành công: {analysis['successful_runs']}")
    print(f"  - Số lần thất bại: {analysis['failed_runs']}")
    print(f"  - Tỷ lệ thành công: {analysis['success_rate']:.1f}%")
    
    if analysis['best_run']:
        print(f"\n🏆 Best Run:")
        print(f"  - Run ID: {analysis['best_run']['run_id']}")
        print(f"  - Config: {analysis['best_run']['config']}")
        print(f"  - Thời gian: {analysis['best_run']['elapsed_time']:.1f}s")
        if analysis['best_run']['test_accuracy'] != 'N/A':
            print(f"  - Test Accuracy: {analysis['best_run']['test_accuracy']:.4f}")
    
    if analysis['config_analysis']:
        print(f"\n⚙️ Phân tích Config:")
        for config_key, stats in analysis['config_analysis'].items():
            print(f"  - {config_key}:")
            print(f"    + Số lần chạy: {stats['count']}")
            print(f"    + Thời gian TB: {stats['avg_time']:.1f}s")
    
    if analysis['recommendations']:
        print(f"\n💡 Khuyến nghị:")
        for rec in analysis['recommendations']:
            print(f"  {rec}")
    
    print("\n" + "=" * 80)


def save_analysis(analysis: Dict, output_file: Path):
    """Lưu phân tích vào file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ Phân tích đã lưu vào: {output_file}")


def main():
    """Main function"""
    results_dir = Path("results/auto_train_10x")
    
    if not results_dir.exists():
        print(f"❌ Không tìm thấy thư mục kết quả: {results_dir}")
        print("   Training có thể chưa hoàn thành hoặc đang chạy...")
        return
    
    # Load results
    summary = load_results(results_dir)
    if not summary:
        print(f"❌ Không tìm thấy file summary.json trong {results_dir}")
        return
    
    # Analyze
    analysis = analyze_results(summary)
    
    # Print
    print_analysis(analysis)
    
    # Save
    analysis_file = results_dir / "analysis.json"
    save_analysis(analysis, analysis_file)
    
    # Create markdown report
    create_markdown_report(analysis, results_dir / "ANALYSIS_REPORT.md")


def create_markdown_report(analysis: Dict, output_file: Path):
    """Tạo báo cáo markdown"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 📊 BÁO CÁO PHÂN TÍCH KẾT QUẢ TRAINING\n\n")
        f.write(f"**Ngày**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 📈 Tổng quan\n\n")
        f.write(f"- **Tổng số lần chạy**: {analysis['total_runs']}\n")
        f.write(f"- **Số lần thành công**: {analysis['successful_runs']}\n")
        f.write(f"- **Số lần thất bại**: {analysis['failed_runs']}\n")
        f.write(f"- **Tỷ lệ thành công**: {analysis['success_rate']:.1f}%\n\n")
        
        if analysis['best_run']:
            f.write("## 🏆 Best Run\n\n")
            f.write(f"- **Run ID**: {analysis['best_run']['run_id']}\n")
            f.write(f"- **Config**: {json.dumps(analysis['best_run']['config'], indent=2)}\n")
            f.write(f"- **Thời gian**: {analysis['best_run']['elapsed_time']:.1f}s\n")
            if analysis['best_run']['test_accuracy'] != 'N/A':
                f.write(f"- **Test Accuracy**: {analysis['best_run']['test_accuracy']:.4f}\n")
            f.write("\n")
        
        if analysis['recommendations']:
            f.write("## 💡 Khuyến nghị\n\n")
            for rec in analysis['recommendations']:
                f.write(f"- {rec}\n")
            f.write("\n")
        
        f.write("---\n\n")
        f.write("**Status**: ✅ Analysis Complete\n")
    
    print(f"✅ Báo cáo markdown đã lưu vào: {output_file}")


if __name__ == "__main__":
    main()






