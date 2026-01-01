# 🖥️ Local Training Guide

## Quick Start

### 1. Setup Environment
```bash
cd training_experiments
pip install -r requirements.txt
```

### 2. Prepare Data
Your data should be in `data/processed/` with this structure:
```
data/processed/
├── train/
│   ├── emotion_0/
│   ├── emotion_1/
│   └── ...
└── test/
    ├── emotion_0/
    ├── emotion_1/
    └── ...
```

### 3. Run Training
```bash
python train_local.py
```

### 4. Monitor Progress
Training will show:
- Progress bar for each epoch
- Train/Test loss and accuracy
- Auto-save best model

### 5. Results
After training completes:
- Model: `checkpoints/local/best_model.pth`
- Results: `checkpoints/local/training_results.json`

---

## Advanced Options

### Custom Configuration
Edit these variables in `train_local.py`:
```python
BATCH_SIZE = 32         # Adjust based on GPU memory
EPOCHS = 50             # Number of training epochs
LEARNING_RATE = 0.001   # Learning rate
```

### GPU/CPU Selection
Automatically uses GPU if available, otherwise CPU:
```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Resume Training
To resume from checkpoint:
```python
checkpoint = torch.load('checkpoints/local/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

## Comparison: Local vs Kaggle

| Feature | Local | Kaggle |
|---------|-------|--------|
| Setup | Manual | Pre-configured |
| GPU | Your GPU | P100 (30h/week free) |
| Datasets | Manual download | One-click add |
| Speed | Depends on GPU | Fast (P100) |
| Best For | Testing, small data | Production training |

---

## Troubleshooting

### Out of Memory
```python
BATCH_SIZE = 16  # Reduce batch size
```

### No GPU
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Missing Data
```bash
# Check data directory
ls data/processed/train
ls data/processed/test
```

---

## Next Steps After Training

1. **Evaluate Model**
   ```bash
   python scripts/evaluate_model.py
   ```

2. **Convert to ONNX**
   ```bash
   python scripts/convert_to_onnx.py
   ```

3. **Test Predictions**
   ```bash
   python scripts/predict_test.py
   ```

4. **Deploy to Edge App**
   ```bash
   # Copy ONNX model to edge app
   cp models/multitask_model.onnx ../ai_edge_app/models/
   ```
