# 🚀 AUTO TRAINING OPTIMIZER - HƯỚNG DẪN

## 📋 Tổng quan

Auto Training Optimizer tự động chạy training nhiều lần với các hyperparameters khác nhau, tối ưu dựa trên kết quả để đạt kết quả cao nhất.

## ✨ Tính năng

- ⚡ **Tự động chạy 10 lần training** với config khác nhau
- ⚡ **Tự động tối ưu** config dựa trên kết quả trước
- ⚡ **Lưu kết quả** tất cả các lần chạy
- ⚡ **Tìm best model** và best config
- ⚡ **So sánh kết quả** giữa các runs

## 🎯 Các Config được Test

1. **Base + MixUp + CutMix**: Config cơ bản với augmentation
2. **High LR**: Learning rate cao (2e-3)
3. **Low LR**: Learning rate thấp (5e-4)
4. **Large Batch**: Batch size lớn (48)
5. **Small Batch**: Batch size nhỏ (16)
6. **High Dropout**: Dropout cao (0.5)
7. **Low Dropout**: Dropout thấp (0.2)
8. **High Age Weight**: Tăng trọng số age (0.7)
9. **Low Age Weight**: Giảm trọng số age (0.3)
10. **Optimal Tuned**: Config tối ưu dựa trên kết quả 9 runs trước

## 🚀 Cách chạy

### Option 1: Dùng Script (Khuyến nghị)

```bash
cd training_experiments
run_auto_training.bat
```

### Option 2: Chạy trực tiếp

```bash
cd training_experiments
python train_10x_automated.py
```

## 📊 Kết quả

Sau khi chạy xong, kết quả được lưu trong `training_results/`:

```
training_results/
├── all_results.json          # Tất cả kết quả
├── best_config.json          # Config tốt nhất
├── run_1_base_mixup_cutmix/  # Run 1
│   ├── best_model.pth
│   ├── metrics.json
│   └── output.log
├── run_2_high_lr/            # Run 2
│   └── ...
└── ...
```

## 📈 Đọc kết quả

### all_results.json
```json
[
  {
    "run_id": 1,
    "config": {...},
    "emotion_acc": 78.5,
    "gender_acc": 94.2,
    "age_mae": 3.8,
    "final_loss": 0.45,
    "success": true
  },
  ...
]
```

### best_config.json
```json
{
  "config": {
    "batch_size": 32,
    "lr": 1e-3,
    "dropout_rate": 0.3,
    ...
  },
  "metrics": {
    "emotion_acc": 79.2,
    "gender_acc": 94.5,
    "age_mae": 3.6
  }
}
```

## ⏱️ Thời gian

- **Mỗi run**: ~1-1.5 giờ (30 epochs)
- **Tổng thời gian**: ~10-15 giờ (10 runs)
- **Tùy thuộc**: Hardware (CPU/GPU), batch size, số epochs

## 💡 Tips

1. **Chạy qua đêm**: Training mất nhiều thời gian, nên chạy qua đêm
2. **Monitor**: Có thể xem TensorBoard logs trong mỗi run directory
3. **Resume**: Nếu bị gián đoạn, có thể chạy lại với `--num_runs` nhỏ hơn
4. **Best Model**: Model tốt nhất ở `training_results/run_<best_id>_*/best_model.pth`

## 🔧 Tùy chỉnh

### Thay đổi số runs
```python
# Trong auto_training_optimizer.py
configs = self.generate_configs()  # Sửa để tạo nhiều config hơn
```

### Thay đổi epochs mỗi run
```python
# Trong generate_configs()
base_config = {
    "epochs": 30,  # Thay đổi ở đây
    ...
}
```

### Thêm config mới
```python
# Trong generate_configs()
configs.append({
    **base_config,
    "run_id": 11,
    "name": "custom_config",
    "lr": 1.5e-3,  # Custom parameters
    ...
})
```

## 📝 Notes

- Mỗi run sẽ tạo thư mục riêng
- Best model được lưu tự động
- Metrics được lưu vào JSON
- Có thể so sánh kết quả giữa các runs

---

**Version**: 1.0  
**Last Updated**: 2025-12-30

