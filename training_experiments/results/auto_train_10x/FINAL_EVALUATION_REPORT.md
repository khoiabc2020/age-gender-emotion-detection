# 📊 BÁO CÁO ĐÁNH GIÁ CUỐI CÙNG - TRAINING 10 LẦN

**Ngày**: 2025-12-30 13:49:18

---

## 📈 Tổng quan

- **Tổng số lần chạy**: 10
- **Số lần thành công**: 0
- **Số lần thất bại**: 10
- **Tỷ lệ thành công**: 0.0%

## 📋 Chi tiết từng Run

### Run 1

- **Config**: {
  "data_dir": "data/processed",
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.001,
  "use_distillation": true,
  "use_qat": false,
  "output_dir": "models/run_1"
}
- **Status**: ❌ Failed
- **Thời gian**: 17.3s

### Run 2

- **Config**: {
  "data_dir": "data/processed",
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.002,
  "use_distillation": true,
  "use_qat": false,
  "output_dir": "models/run_2"
}
- **Status**: ❌ Failed
- **Thời gian**: 11.3s

### Run 3

- **Config**: {
  "data_dir": "data/processed",
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.0005,
  "use_distillation": true,
  "use_qat": false,
  "output_dir": "models/run_3"
}
- **Status**: ❌ Failed
- **Thời gian**: 10.5s

### Run 4

- **Config**: {
  "data_dir": "data/processed",
  "epochs": 50,
  "batch_size": 64,
  "learning_rate": 0.001,
  "use_distillation": true,
  "use_qat": false,
  "output_dir": "models/run_4"
}
- **Status**: ❌ Failed
- **Thời gian**: 10.7s

### Run 5

- **Config**: {
  "data_dir": "data/processed",
  "epochs": 50,
  "batch_size": 16,
  "learning_rate": 0.001,
  "use_distillation": true,
  "use_qat": false,
  "output_dir": "models/run_5"
}
- **Status**: ❌ Failed
- **Thời gian**: 12.6s

### Run 6

- **Config**: {
  "data_dir": "data/processed",
  "epochs": 75,
  "batch_size": 32,
  "learning_rate": 0.001,
  "use_distillation": true,
  "use_qat": false,
  "output_dir": "models/run_6"
}
- **Status**: ❌ Failed
- **Thời gian**: 9.9s

### Run 7

- **Config**: {
  "data_dir": "data/processed",
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.001,
  "use_distillation": true,
  "use_qat": true,
  "output_dir": "models/run_7"
}
- **Status**: ❌ Failed
- **Thời gian**: 9.5s

### Run 8

- **Config**: {
  "data_dir": "data/processed",
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.0005,
  "use_distillation": true,
  "use_qat": true,
  "output_dir": "models/run_8"
}
- **Status**: ❌ Failed
- **Thời gian**: 9.4s

### Run 9

- **Config**: {
  "data_dir": "data/processed",
  "epochs": 50,
  "batch_size": 64,
  "learning_rate": 0.002,
  "use_distillation": true,
  "use_qat": false,
  "output_dir": "models/run_9"
}
- **Status**: ❌ Failed
- **Thời gian**: 9.6s

### Run 10

- **Config**: {
  "data_dir": "data/processed",
  "epochs": 60,
  "batch_size": 48,
  "learning_rate": 0.0015,
  "use_distillation": true,
  "use_qat": false,
  "output_dir": "models/run_10"
}
- **Status**: ❌ Failed
- **Thời gian**: 10.2s

---

## 💡 Kết luận

⚠️ **Tất cả lần training đều thất bại.**

**Nguyên nhân có thể:**
1. Thiếu dữ liệu training
2. Lỗi trong script training
3. Thiếu dependencies
4. Lỗi cấu hình

**Giải pháp:**
1. Kiểm tra dữ liệu: `python scripts/check_datasets.py`
2. Kiểm tra log: Xem `results/auto_train_10x/run_*_results.json`
3. Chạy thử 1 lần: `python train_week2_lightweight.py --data_dir data/processed --epochs 1`

---

**Status**: ✅ Report Complete
