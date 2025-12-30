# 📊 PHÂN TÍCH KẾT QUẢ TRAINING

**Ngày**: 2025-12-30  
**Script**: `train_10x_automated.py`  
**Số lần chạy**: 10

---

## 📋 TỔNG QUAN

Script tự động chạy training 10 lần với các hyperparameters khác nhau để tìm config tối ưu nhất.

---

## 🎯 CÁC CONFIG ĐƯỢC TEST

1. **Base**: LR=0.001, Batch=32, Epochs=50, Distillation=True
2. **High LR**: LR=0.002
3. **Low LR**: LR=0.0005
4. **Large Batch**: Batch=64
5. **Small Batch**: Batch=16
6. **More Epochs**: Epochs=75
7. **With QAT**: QAT=True
8. **Low LR + QAT**: LR=0.0005, QAT=True
9. **Large Batch + High LR**: Batch=64, LR=0.002
10. **Optimal**: LR=0.0015, Batch=48, Epochs=60

---

## 📊 KẾT QUẢ

### Xem kết quả chi tiết:
```bash
# Xem summary
cat results/auto_train_10x/summary.json

# Xem từng run
cat results/auto_train_10x/run_*_results.json
```

### Metrics được đánh giá:
- **Emotion Accuracy**: % chính xác nhận diện cảm xúc
- **Gender Accuracy**: % chính xác nhận diện giới tính
- **Age MAE**: Mean Absolute Error cho tuổi (càng thấp càng tốt)
- **Final Loss**: Loss cuối cùng

---

## 🏆 BEST MODEL

Best model sẽ được tự động chọn dựa trên:
- Emotion Accuracy (weight: 0.4)
- Gender Accuracy (weight: 0.3)
- Age MAE (weight: 0.3)

**Location**: `results/auto_train_10x/run_<best_id>_*/best_model.pth`

---

## 📈 SO SÁNH KẾT QUẢ

Sau khi training xong, so sánh các config để tìm:
- Config cho Emotion Accuracy cao nhất
- Config cho Gender Accuracy cao nhất
- Config cho Age MAE thấp nhất
- Config cân bằng tốt nhất (best overall)

---

## 🔄 SỬ DỤNG BEST MODEL

```bash
# Copy best model
copy results\auto_train_10x\run_<best_id>\best_model.pth models\best_model.pth

# Convert to ONNX
python scripts/convert_to_onnx.py \
    --model_path models/best_model.pth \
    --output_path ai_edge_app/models/multitask_efficientnet.onnx
```

---

**Status**: 🔄 Training đang chạy...

