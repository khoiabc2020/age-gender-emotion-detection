"""
Automated Training Script - 10 lần training
Tối ưu và chạy training 10 lần để đạt kết quả tốt nhất
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import subprocess
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
training_dir = Path(__file__).parent
sys.path.insert(0, str(training_dir / "src"))

try:
    from src.utils.logging import setup_logger
    logger = setup_logger('AutoTrain', str(training_dir / 'logs' / 'auto_train.log'))
except ImportError:
    # Fallback to basic logging if utils module not available
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(training_dir / 'logs' / 'auto_train.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('AutoTrain')


def run_training(run_id: int, config: dict) -> dict:
    """
    Run single training experiment
    
    Args:
        run_id: Training run ID
        config: Training configuration
        
    Returns:
        Results dictionary
    """
    logger.info(f"🚀 Starting training run {run_id}/10")
    logger.info(f"Config: {config}")
    
    start_time = time.time()
    
    # Prepare command
    train_script = project_root / "training_experiments" / "train_week2_lightweight.py"
    
    cmd = [
        sys.executable,
        str(train_script),
        "--data_dir", str(config.get('data_dir', 'data/processed')),
        "--epochs", str(config.get('epochs', 50)),
        "--batch_size", str(config.get('batch_size', 32)),
        "--lr", str(config.get('learning_rate', 0.001)),  # Fixed: --lr instead of --learning_rate
        "--save_dir", str(config.get('output_dir', f'checkpoints/run_{run_id}'))  # Fixed: --save_dir instead of --output_dir
    ]
    
    if config.get('use_distillation', False):
        cmd.append("--use_distillation")
    if config.get('use_qat', False):
        cmd.append("--use_qat")
    
    try:
        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 2  # 2 hours timeout
        )
        
        elapsed_time = time.time() - start_time
        
        # Parse results from output
        output = result.stdout + result.stderr
        
        # Log output for debugging if failed
        if result.returncode != 0:
            logger.warning(f"Training run {run_id} failed with returncode {result.returncode}")
            # Save full output to file for debugging
            error_log_file = results_dir / f"run_{run_id}_error.log"
            with open(error_log_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"ERROR LOG - Run {run_id}\n")
                f.write("=" * 80 + "\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
            logger.warning(f"Full error log saved to: {error_log_file}")
            if result.stdout:
                logger.warning(f"STDOUT (first 1000 chars): {result.stdout[:1000]}")
            if result.stderr:
                logger.warning(f"STDERR (first 1000 chars): {result.stderr[:1000]}")
        
        # Extract metrics (simplified - adjust based on actual output format)
        metrics = {
            'run_id': run_id,
            'config': config,
            'success': result.returncode == 0,
            'elapsed_time': elapsed_time,
            'returncode': result.returncode,
            'stdout': result.stdout[:2000] if result.returncode != 0 else None,  # Save error output for debugging
            'stderr': result.stderr[:2000] if result.returncode != 0 else None
        }
        
        # Try to extract final metrics from output
        if 'test_accuracy' in output.lower():
            # Parse test accuracy if available
            for line in output.split('\n'):
                if 'test' in line.lower() and 'accuracy' in line.lower():
                    try:
                        # Extract number
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'accuracy' in part.lower() and i + 1 < len(parts):
                                metrics['test_accuracy'] = float(parts[i+1])
                                break
                    except:
                        pass
        
        logger.info(f"✅ Training run {run_id} completed in {elapsed_time:.1f}s")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Training run {run_id} timed out")
        return {
            'run_id': run_id,
            'config': config,
            'success': False,
            'error': 'timeout'
        }
    except Exception as e:
        logger.error(f"❌ Training run {run_id} failed: {e}")
        return {
            'run_id': run_id,
            'config': config,
            'success': False,
            'error': str(e)
        }


def generate_configs() -> list:
    """
    Generate 10 different training configurations
    Vary hyperparameters to find best combination
    """
    # Auto-detect data directory
    from pathlib import Path
    if Path('data/processed/train').exists():
        data_dir = 'data/processed'
    elif Path('data/processed/utkface/train').exists():
        data_dir = 'data/processed/utkface'
    else:
        data_dir = 'data/processed'  # Default
    
    base_config = {
        'data_dir': data_dir,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'use_distillation': True,
        'use_qat': False,
        'output_dir': 'models'
    }
    
    configs = []
    
    # Config 1: Base
    configs.append(base_config.copy())
    
    # Config 2: Higher LR
    config = base_config.copy()
    config['learning_rate'] = 0.002
    configs.append(config)
    
    # Config 3: Lower LR
    config = base_config.copy()
    config['learning_rate'] = 0.0005
    configs.append(config)
    
    # Config 4: Larger batch
    config = base_config.copy()
    config['batch_size'] = 64
    configs.append(config)
    
    # Config 5: Smaller batch
    config = base_config.copy()
    config['batch_size'] = 16
    configs.append(config)
    
    # Config 6: More epochs
    config = base_config.copy()
    config['epochs'] = 75
    configs.append(config)
    
    # Config 7: With QAT
    config = base_config.copy()
    config['use_qat'] = True
    configs.append(config)
    
    # Config 8: Lower LR + QAT
    config = base_config.copy()
    config['learning_rate'] = 0.0005
    config['use_qat'] = True
    configs.append(config)
    
    # Config 9: Larger batch + Higher LR
    config = base_config.copy()
    config['batch_size'] = 64
    config['learning_rate'] = 0.002
    configs.append(config)
    
    # Config 10: Optimal (based on previous results)
    config = base_config.copy()
    config['learning_rate'] = 0.0015
    config['batch_size'] = 48
    config['epochs'] = 60
    configs.append(config)
    
    return configs


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("🚀 AUTOMATED TRAINING - 10 LẦN")
    print("=" * 80)
    
    # Create results directory
    results_dir = project_root / "training_experiments" / "results" / "auto_train_10x"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate configurations
    configs = generate_configs()
    
    # Optimize configs for faster testing (reduce epochs for initial runs)
    # User can increase epochs later for full training
    print("\n⚙️  Optimizing configs for efficient training...")
    print("   Note: Epochs reduced to 5 for quick testing. Edit train_10x_automated.py to increase.")
    for config in configs:
        # Reduce epochs for faster testing - user can increase later
        original_epochs = config.get('epochs', 50)
        config['epochs'] = 5  # Quick test with 5 epochs
        config['original_epochs'] = original_epochs  # Save original for reference
        # Reduce batch size if too large for CPU
        if config.get('batch_size', 32) > 32:
            config['batch_size'] = min(32, config.get('batch_size', 64))
    
    # Run training 10 times
    all_results = []
    
    for i, config in enumerate(configs, 1):
        config['output_dir'] = f'models/run_{i}'
        print(f"\n{'='*60}")
        print(f"Run {i}/10: epochs={config['epochs']}, batch_size={config['batch_size']}, lr={config['learning_rate']}")
        print(f"{'='*60}")
        result = run_training(i, config)
        all_results.append(result)
        
        # Save intermediate results
        results_file = results_dir / f"run_{i}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Print progress
        if result.get('success'):
            print(f"✅ Run {i} completed successfully")
        else:
            print(f"❌ Run {i} failed - check error log")
        
        # Small delay between runs
        if i < len(configs):
            time.sleep(2)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_runs': len(all_results),
        'successful_runs': sum(1 for r in all_results if r.get('success', False)),
        'results': all_results,
        'best_run': None
    }
    
    # Find best run
    successful_results = [r for r in all_results if r.get('success', False)]
    if successful_results:
        # Sort by test_accuracy if available
        if any('test_accuracy' in r for r in successful_results):
            best = max(successful_results, key=lambda x: x.get('test_accuracy', 0))
        else:
            # Sort by elapsed time (faster is better if same accuracy)
            best = min(successful_results, key=lambda x: x.get('elapsed_time', float('inf')))
        summary['best_run'] = best
    
    summary_file = results_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TỔNG KẾT TRAINING 10 LẦN")
    print("=" * 80)
    
    print(f"\nTổng số lần chạy: {len(all_results)}")
    print(f"Số lần thành công: {summary['successful_runs']}")
    print(f"Số lần thất bại: {len(all_results) - summary['successful_runs']}")
    
    if summary['best_run']:
        print(f"\n🏆 Best Run: Run {summary['best_run']['run_id']}")
        print(f"   Config: {summary['best_run']['config']}")
        if 'test_accuracy' in summary['best_run']:
            print(f"   Test Accuracy: {summary['best_run']['test_accuracy']:.4f}")
        print(f"   Elapsed Time: {summary['best_run']['elapsed_time']:.1f}s")
    
    print(f"\n📁 Results saved to: {results_dir}")
    print("=" * 80 + "\n")
    
    return summary


if __name__ == "__main__":
    try:
        summary = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)

