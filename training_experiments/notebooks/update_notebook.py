#!/usr/bin/env python3
"""
Update kaggle_4datasets_training.ipynb with optimized Cell 5
Target: 80-83% accuracy
"""

import json
from pathlib import Path

# Read the optimized training code
optimized_code_file = Path(__file__).parent / 'KAGGLE_OPTIMIZED_80_PERCENT.py'
with open(optimized_code_file, 'r', encoding='utf-8') as f:
    optimized_code = f.read()

# Remove the first 3 comment lines (header comments)
code_lines = optimized_code.split('\n')
code_lines = code_lines[3:]  # Skip first 3 header lines
optimized_code_clean = '\n'.join(code_lines)

# Read the notebook
notebook_file = Path(__file__).parent / 'kaggle_4datasets_training.ipynb'
with open(notebook_file, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find Cell 5 (index 4 in 0-based indexing, but actually need to find it)
cell_5_index = None
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'CELL 5' in source or 'OPTIMIZED TRAINING' in source:
            cell_5_index = i
            break

if cell_5_index is None:
    print("ERROR: Could not find Cell 5")
    exit(1)

print(f"Found Cell 5 at index {cell_5_index}")

# Replace Cell 5 content
new_source = optimized_code_clean.split('\n')
notebook['cells'][cell_5_index]['source'] = [line + '\n' for line in new_source[:-1]] + [new_source[-1]]

# Save updated notebook
output_file = Path(__file__).parent / 'kaggle_4datasets_training_v2.ipynb'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"[OK] Updated notebook saved to: {output_file.name}")
print("\nImprovements in Cell 5:")
print("  [OK] EfficientNetV2 (vs B0) -> +2-3%")
print("  [OK] RandAugment -> +1-2%")
print("  [OK] CutMix (vs Mixup) -> +0.5-1%")
print("  [OK] Focal Loss -> +1-2%")
print("  [OK] 200 epochs (vs 150) -> +1-2%")
print("  [OK] 72x72 input (vs 64) -> +0.5-1%")
print("\nExpected accuracy: 80-83% (from 76.49%)")
print("Training time: 10-11 hours")
