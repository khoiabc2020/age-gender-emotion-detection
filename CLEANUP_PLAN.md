# PROJECT CLEANUP PLAN

## 📋 **PHÂN TÍCH:**

### **ROOT LEVEL - Markdown Files (10 files):**
```
✅ KEEP:
- README.md                          (main project readme)
- CONTRIBUTING.md                    (contribution guide)
- PROJECT_DOCUMENTATION.md           (main documentation)

❌ DELETE (OUTDATED/DUPLICATE):
- PRODUCTION_READY.md                (outdated status)
- PRODUCTION_TODO.md                 (outdated todos)
- PRODUCTION_TRAINING_READY.md       (outdated)
- TRAINING_RESULTS.md                (old, use training_experiments/)
- HUONG_DAN_HOC_TAP_VA_SU_DUNG.md   (huge, redundant with docs/)
```

### **training_experiments/ - Markdown Files (18 files!):**
```
✅ KEEP (ESSENTIAL):
- README.md                                    (folder readme)
- POST_TRAINING_WORKFLOW.md                   (complete guide)
- TRAINING_VERSIONS_COMPARISON.md             (version comparison)
- TRAINING_SUCCESS_76.49.md                   (training report)

❌ DELETE (OUTDATED/DUPLICATE):
- NOTEBOOK_UPGRADE_COMPLETE.md                (temporary status)
- POST_TRAINING_QUICK_REF.md                  (duplicate of workflow)
- QUICK_ACTION_GUIDE.md                       (duplicate)
- COLAB_PRODUCTION_TRAINING.md                (old, superseded)
- AUTO_TRAINING_GUIDE.md                      (not used)
- DATASETS_INFO.md                            (outdated)
- TRAINING_RESULTS_ANALYSIS.md                (old analysis)

- notebooks/NOTEBOOK_UPGRADED_GUIDE.md        (temporary)
- notebooks/TRAINING_OPTIMIZATION_GUIDE.md    (duplicate info)
- notebooks/PRODUCTION_CELLS_UPDATE.md        (old)
- notebooks/KAGGLE_TRAINING_GUIDE.md          (old)
- notebooks/KAGGLE_OPTIMIZED_COMPLETE.md      (old)
- notebooks/KAGGLE_IMPROVED_TRAINING.md       (old)
- notebooks/KAGGLE_4DATASETS_COMPLETE.md      (old)
- notebooks/FREE_GPU_ALTERNATIVES.md          (not needed)
- notebooks/DATASET_ALTERNATIVES.md           (outdated)
- notebooks/ALTERNATIVE_DATASETS_VERIFIED.md  (duplicate)

- results/auto_train_10x/ANALYSIS_REPORT.md   (old experiment)
- results/auto_train_10x/FINAL_EVALUATION_REPORT.md (old)
```

### **docs/ - Keep Organized (9 files):**
```
✅ KEEP ALL:
- ROADMAP.md
- PRODUCTION_ROADMAP.md
- MLOPS_ROADMAP.md
- OPTIMIZATION.md
- SECURITY.md
- CI_CD.md
- SETUP.md
- GIT_GUIDE.md
- GITHUB_AND_COLAB_GUIDE.md (consolidate?)
- PROJECT_DETAILS.md
```

### **ai_edge_app/ - App Specific (3 files):**
```
✅ KEEP:
- README.md                 (app readme)

❌ DELETE (OUTDATED):
- ULTIMATE_ROADMAP.md       (old roadmap)
- YOLO_COMPLETE.md          (old feature)
```

### **Python Scripts - Training (many!):**
```
✅ KEEP (CORE):
- train_10x_automated.py              (automated training)
- analyze_results.py                  (result analysis)
- update_results_and_evaluate.py      (evaluation)
- test_new_model.py                   (testing)

❌ DELETE (OUTDATED/OLD VERSIONS):
- train_production.py                 (old version)
- train_production_full.py            (old version)
- train_colab_simple.py               (superseded by notebook)
- train_week2_lightweight.py          (old experiment)
- prepare_and_train.py                (old)
```

### **Scripts Folder:**
```
✅ KEEP:
- scripts/push_to_github.bat
- scripts/check_week1_requirements.py
- scripts/convert_to_onnx.py
- scripts/prepare_fer2013.py

❌ DELETE (OLD):
- All old experiment scripts
```

### **Checkpoints & Results:**
```
✅ KEEP:
- checkpoints/production/best_model.pth
- results/latest_training_results.json

❌ DELETE:
- checkpoints/quick_train/           (old experiments)
- checkpoints/week2/                 (old experiments)
- training_results/run_1-10/         (old experiments)
- results/auto_train_10x/            (old experiments)
```

---

## 🎯 **CLEANUP ACTIONS:**

### **Total Files to Delete: ~35 files**
### **Space Saved: ~500 MB (checkpoints) + many markdown files**

---

## ✅ **NEW STRUCTURE (CLEAN):**

```
project/
├── README.md                          ✅ Main
├── CONTRIBUTING.md                    ✅ Guide
├── PROJECT_DOCUMENTATION.md           ✅ Documentation
├── test_new_model.py                  ✅ Testing
│
├── docs/                              ✅ All documentation
│   ├── ROADMAP.md
│   ├── PRODUCTION_ROADMAP.md
│   ├── MLOPS_ROADMAP.md
│   ├── OPTIMIZATION.md
│   ├── SECURITY.md
│   ├── CI_CD.md
│   ├── SETUP.md
│   └── GIT_GUIDE.md
│
├── training_experiments/              ✅ Training only
│   ├── README.md
│   ├── POST_TRAINING_WORKFLOW.md      ✅ Complete guide
│   ├── TRAINING_VERSIONS_COMPARISON.md
│   ├── TRAINING_SUCCESS_76.49.md
│   ├── train_10x_automated.py
│   ├── analyze_results.py
│   ├── update_results_and_evaluate.py
│   ├── notebooks/
│   │   ├── kaggle_4datasets_training.ipynb  ✅ Main notebook
│   │   ├── KAGGLE_OPTIMIZED_80_PERCENT.py
│   │   └── ADVANCED_TRAINING_IMPROVEMENTS.py
│   ├── checkpoints/
│   │   └── production/best_model.pth  ✅ Only keep latest
│   ├── results/
│   │   └── latest_training_results.json
│   └── scripts/                       ✅ Essential scripts only
│
├── ai_edge_app/                       ✅ Application
│   ├── README.md
│   ├── main.py
│   └── src/
│
├── backend_api/                       ✅ API
├── dashboard/                         ✅ Dashboard
├── database/                          ✅ Database
├── k8s/                              ✅ Kubernetes
└── scripts/                          ✅ Build scripts
```

---

## 🗑️ **FILES TO DELETE:**

### Root:
- PRODUCTION_READY.md
- PRODUCTION_TODO.md
- PRODUCTION_TRAINING_READY.md
- TRAINING_RESULTS.md
- HUONG_DAN_HOC_TAP_VA_SU_DUNG.md

### training_experiments/:
- NOTEBOOK_UPGRADE_COMPLETE.md
- POST_TRAINING_QUICK_REF.md
- QUICK_ACTION_GUIDE.md
- COLAB_PRODUCTION_TRAINING.md
- AUTO_TRAINING_GUIDE.md
- DATASETS_INFO.md
- TRAINING_RESULTS_ANALYSIS.md
- train_production.py
- train_production_full.py
- train_colab_simple.py
- train_week2_lightweight.py
- prepare_and_train.py
- CHAY_TU_DONG_COLAB.bat
- run_auto_training.bat

### training_experiments/notebooks/:
- NOTEBOOK_UPGRADED_GUIDE.md
- TRAINING_OPTIMIZATION_GUIDE.md
- PRODUCTION_CELLS_UPDATE.md
- KAGGLE_TRAINING_GUIDE.md
- KAGGLE_OPTIMIZED_COMPLETE.md
- KAGGLE_IMPROVED_TRAINING.md
- KAGGLE_4DATASETS_COMPLETE.md
- FREE_GPU_ALTERNATIVES.md
- DATASET_ALTERNATIVES.md
- ALTERNATIVE_DATASETS_VERIFIED.md
- train_on_colab_auto.ipynb (old version)
- train_production_colab.ipynb (old version)

### training_experiments/checkpoints/:
- checkpoints/quick_train/
- checkpoints/week2/

### training_experiments/training_results/:
- All run_1 to run_10 folders

### training_experiments/results/:
- results/auto_train_10x/

### ai_edge_app/:
- ULTIMATE_ROADMAP.md
- YOLO_COMPLETE.md

---

**Total: ~35 files + old checkpoints**
