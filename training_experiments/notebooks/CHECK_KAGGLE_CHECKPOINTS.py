# RESUME TRAINING FROM CHECKPOINT
# Nếu Kaggle có auto-save checkpoint, dùng code này để continue

import torch
from pathlib import Path

# Check available checkpoints
checkpoint_dir = Path('/kaggle/working/checkpoints_production')
output_dir = Path('/kaggle/output')

print("="*60)
print("CHECKING FOR SAVED CHECKPOINTS")
print("="*60)

# Check /kaggle/working/
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    if checkpoints:
        print(f"\n[FOUND] {len(checkpoints)} checkpoints in /kaggle/working/:")
        for cp in checkpoints:
            size = cp.stat().st_size / (1024*1024)
            print(f"  ✓ {cp.name} ({size:.1f} MB)")
    else:
        print("\n[EMPTY] No checkpoints in /kaggle/working/")
else:
    print("\n[NOT FOUND] /kaggle/working/checkpoints_production/")

# Check /kaggle/output/ (persistent)
if output_dir.exists():
    checkpoints = list(output_dir.glob('*.pth'))
    if checkpoints:
        print(f"\n[FOUND] {len(checkpoints)} checkpoints in /kaggle/output/:")
        for cp in checkpoints:
            size = cp.stat().st_size / (1024*1024)
            print(f"  ✓ {cp.name} ({size:.1f} MB)")
    else:
        print("\n[EMPTY] No checkpoints in /kaggle/output/")
else:
    print("\n[NOT FOUND] /kaggle/output/")

# Check results
results_files = []
for loc in [checkpoint_dir, output_dir]:
    if loc.exists():
        results_files.extend(list(loc.glob('*.json')))

if results_files:
    print(f"\n[FOUND] {len(results_files)} result files:")
    for rf in results_files:
        print(f"  ✓ {rf}")
else:
    print("\n[NOT FOUND] No result files")

print("="*60)

# If checkpoint found, can resume training
# If not found, need to train from scratch
