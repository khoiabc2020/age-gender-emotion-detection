"""
Script tự động setup và chạy training trên Colab
Tạo notebook tự động chạy với tất cả các bước
"""

import json
from pathlib import Path


def create_auto_run_notebook():
    """Tạo notebook tự động chạy"""
    
    # Đọc notebook hiện tại
    notebook_path = Path(__file__).parent.parent / 'notebooks' / 'train_on_colab.ipynb'
    
    # Tạo notebook mới với auto-run
    notebook = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    # Cell 1: Markdown - Hướng dẫn
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 Training Tự Động trên Google Colab\n",
            "\n",
            "Notebook này sẽ tự động:\n",
            "1. ✅ Cài đặt dependencies\n",
            "2. ✅ Kiểm tra GPU\n",
            "3. ✅ Mount Google Drive\n",
            "4. ✅ Download code từ Google Drive\n",
            "5. ✅ Download/Setup dữ liệu\n",
            "6. ✅ Chạy training tự động\n",
            "7. ✅ Lưu kết quả về Google Drive\n",
            "\n",
            "**Lưu ý**: Chọn GPU runtime trước khi chạy!"
        ]
    })
    
    # Cell 2: Cài đặt dependencies
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cài đặt dependencies\n",
            "%pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n",
            "%pip install -q albumentations tqdm tensorboard onnx onnxruntime\n",
            "%pip install -q pandas numpy Pillow opencv-python\n",
            "%pip install -q kagglehub\n",
            "\n",
            "print(\"✅ Đã cài đặt xong các thư viện!\")"
        ]
    })
    
    # Cell 3: Kiểm tra GPU
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.cuda as cuda\n",
            "\n",
            "print(f\"PyTorch version: {torch.__version__}\")\n",
            "print(f\"CUDA available: {cuda.is_available()}\")\n",
            "if cuda.is_available():\n",
            "    print(f\"CUDA version: {cuda.version()}\")\n",
            "    print(f\"GPU device: {cuda.get_device_name(0)}\")\n",
            "    print(f\"GPU memory: {cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\")\n",
            "else:\n",
            "    print(\"⚠️  Không có GPU! Vui lòng chọn GPU runtime:\")\n",
            "    print(\"   Runtime → Change runtime type → Hardware accelerator → GPU\")"
        ]
    })
    
    # Cell 4: Mount Google Drive
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from google.colab import drive\n",
            "import os\n",
            "\n",
            "# Mount Google Drive\n",
            "drive.mount('/content/drive')\n",
            "\n",
            "# Tạo thư mục lưu kết quả\n",
            "DRIVE_RESULTS_DIR = '/content/drive/MyDrive/age_gender_emotion_training'\n",
            "os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)\n",
            "print(f\"✅ Đã mount Google Drive\")\n",
            "print(f\"📁 Kết quả sẽ lưu tại: {DRIVE_RESULTS_DIR}\")"
        ]
    })
    
    # Cell 5: Download code từ Google Drive
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import zipfile\n",
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "# Đường dẫn file zip trên Drive (thay đổi theo vị trí file của bạn)\n",
            "DRIVE_CODE_ZIP = '/content/drive/MyDrive/Colab_Training/training_experiments_*.zip'\n",
            "\n",
            "# Tìm file zip mới nhất\n",
            "import glob\n",
            "zip_files = glob.glob('/content/drive/MyDrive/Colab_Training/training_experiments_*.zip')\n",
            "\n",
            "if zip_files:\n",
            "    # Lấy file mới nhất\n",
            "    latest_zip = max(zip_files, key=os.path.getctime)\n",
            "    print(f\"📦 Tìm thấy file: {os.path.basename(latest_zip)}\")\n",
            "    \n",
            "    # Giải nén\n",
            "    print(\"📂 Đang giải nén...\")\n",
            "    with zipfile.ZipFile(latest_zip, 'r') as zip_ref:\n",
            "        zip_ref.extractall('/content/project')\n",
            "    print(\"✅ Đã giải nén code\")\n",
            "    \n",
            "    # Kiểm tra cấu trúc\n",
            "    project_dir = Path('/content/project/training_experiments')\n",
            "    if project_dir.exists():\n",
            "        print(f\"✅ Code đã sẵn sàng tại: {project_dir}\")\n",
            "    else:\n",
            "        print(\"⚠️  Cần kiểm tra lại cấu trúc thư mục\")\n",
            "else:\n",
            "    print(\"⚠️  Không tìm thấy file zip trên Drive\")\n",
            "    print(\"   Hãy upload file zip vào: /content/drive/MyDrive/Colab_Training/\")"
        ]
    })
    
    # Cell 6: Setup dữ liệu (từ Drive hoặc download)
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import shutil\n",
            "from pathlib import Path\n",
            "\n",
            "# Option 1: Copy từ Google Drive (nếu đã upload)\n",
            "DRIVE_DATA_DIR = '/content/drive/MyDrive/age_gender_emotion_data'\n",
            "LOCAL_DATA_DIR = Path('/content/project/training_experiments/data/processed')\n",
            "\n",
            "if os.path.exists(DRIVE_DATA_DIR):\n",
            "    print(f\"📁 Copy dữ liệu từ Drive...\")\n",
            "    shutil.copytree(DRIVE_DATA_DIR, LOCAL_DATA_DIR, dirs_exist_ok=True)\n",
            "    print(f\"✅ Đã copy dữ liệu\")\n",
            "else:\n",
            "    print(f\"⚠️  Chưa có dữ liệu trên Drive\")\n",
            "    print(f\"   Upload dữ liệu vào: {DRIVE_DATA_DIR}\")\n",
            "    print(f\"   Hoặc sử dụng cell tiếp theo để download từ Kaggle\")"
        ]
    })
    
    # Cell 7: Kiểm tra dữ liệu
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "\n",
            "data_dir = Path('/content/project/training_experiments/data/processed')\n",
            "train_dir = data_dir / 'train'\n",
            "val_dir = data_dir / 'val'\n",
            "test_dir = data_dir / 'test'\n",
            "\n",
            "print(\"📊 Kiểm tra dữ liệu:\")\n",
            "print(f\"   Train: {train_dir.exists()}\")\n",
            "print(f\"   Val: {val_dir.exists()}\")\n",
            "print(f\"   Test: {test_dir.exists()}\")\n",
            "\n",
            "if train_dir.exists():\n",
            "    train_images = list(train_dir.glob('**/*.jpg')) + list(train_dir.glob('**/*.png'))\n",
            "    val_images = list(val_dir.glob('**/*.jpg')) + list(val_dir.glob('**/*.png')) if val_dir.exists() else []\n",
            "    test_images = list(test_dir.glob('**/*.jpg')) + list(test_dir.glob('**/*.png')) if test_dir.exists() else []\n",
            "    \n",
            "    print(f\"\\n📸 Số lượng ảnh:\")\n",
            "    print(f\"   Train: {len(train_images)}\")\n",
            "    print(f\"   Val: {len(val_images)}\")\n",
            "    print(f\"   Test: {len(test_images)}\")\n",
            "    \n",
            "    if len(train_images) == 0:\n",
            "        print(\"\\n⚠️  Chưa có ảnh trong thư mục train!\")\n",
            "else:\n",
            "    print(\"\\n⚠️  Chưa có dữ liệu processed!\")"
        ]
    })
    
    # Cell 8: Chạy training tự động
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "# Thêm project vào Python path\n",
            "project_dir = Path('/content/project/training_experiments')\n",
            "sys.path.insert(0, str(project_dir))\n",
            "os.chdir(project_dir)\n",
            "\n",
            "# Cấu hình training\n",
            "EPOCHS = 50\n",
            "BATCH_SIZE = 32\n",
            "LEARNING_RATE = 1e-3\n",
            "USE_QAT = True\n",
            "USE_DISTILLATION = True\n",
            "\n",
            "print(\"🚀 Bắt đầu training tự động...\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"⚙️  Cấu hình:\")\n",
            "print(f\"   Epochs: {EPOCHS}\")\n",
            "print(f\"   Batch size: {BATCH_SIZE}\")\n",
            "print(f\"   Learning rate: {LEARNING_RATE}\")\n",
            "print(f\"   QAT: {USE_QAT}\")\n",
            "print(f\"   Distillation: {USE_DISTILLATION}\")\n",
            "print(\"=\" * 60)\n",
            "\n",
            "# Build command\n",
            "cmd = f\"python train_week2_lightweight.py --data_dir data/processed --epochs {EPOCHS} --batch_size {BATCH_SIZE} --lr {LEARNING_RATE} --save_dir checkpoints/week2_colab --num_workers 2\"\n",
            "\n",
            "if USE_QAT:\n",
            "    cmd += \" --use_qat\"\n",
            "if USE_DISTILLATION:\n",
            "    cmd += \" --use_distillation\"\n",
            "\n",
            "# Chạy training\n",
            "os.system(cmd)\n",
            "\n",
            "print(\"\\n\" + \"=\" * 60)\n",
            "print(\"✅ Training hoàn tất!\")"
        ]
    })
    
    # Cell 9: Lưu kết quả về Drive
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import shutil\n",
            "from datetime import datetime\n",
            "from pathlib import Path\n",
            "\n",
            "# Tạo thư mục lưu kết quả với timestamp\n",
            "timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n",
            "results_dir = Path(DRIVE_RESULTS_DIR) / f'training_{timestamp}'\n",
            "results_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# Copy checkpoints\n",
            "checkpoint_dir = Path('/content/project/training_experiments/checkpoints/week2_colab')\n",
            "if checkpoint_dir.exists():\n",
            "    print(f\"📦 Copy checkpoints...\")\n",
            "    shutil.copytree(checkpoint_dir, results_dir / 'checkpoints', dirs_exist_ok=True)\n",
            "    print(f\"✅ Đã copy checkpoints\")\n",
            "\n",
            "# Copy logs\n",
            "logs_dir = checkpoint_dir / 'logs'\n",
            "if logs_dir.exists():\n",
            "    print(f\"📊 Copy logs...\")\n",
            "    shutil.copytree(logs_dir, results_dir / 'logs', dirs_exist_ok=True)\n",
            "    print(f\"✅ Đã copy logs\")\n",
            "\n",
            "# Copy ONNX model\n",
            "onnx_file = checkpoint_dir / 'mobileone_multitask.onnx'\n",
            "if onnx_file.exists():\n",
            "    print(f\"📄 Copy ONNX model...\")\n",
            "    shutil.copy2(onnx_file, results_dir / 'mobileone_multitask.onnx')\n",
            "    print(f\"✅ Đã copy ONNX model\")\n",
            "\n",
            "print(f\"\\n✅ Tất cả kết quả đã được lưu tại:\")\n",
            "print(f\"   {results_dir}\")"
        ]
    })
    
    # Lưu notebook
    output_path = Path(__file__).parent.parent / 'notebooks' / 'train_on_colab_auto.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"Da tao notebook tu dong: {output_path}")
    return output_path


if __name__ == "__main__":
    create_auto_run_notebook()

