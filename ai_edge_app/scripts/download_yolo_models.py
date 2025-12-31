"""
Script để download YOLO models cho face/person detection
"""

import os
import urllib.request
from pathlib import Path


def download_file(url: str, output_path: Path):
    """Download file from URL"""
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        return False


def main():
    """Download YOLO models"""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("YOLO Models Downloader")
    print("=" * 60)
    print("\nChọn model để download:")
    print("1. YOLOv8n Face Detection (6MB)")
    print("2. YOLOv8n Person Detection - COCO (6MB)")
    print("3. YOLOv8s Face Detection (22MB) - More accurate")
    print("4. Download tất cả")
    
    choice = input("\nNhập lựa chọn (1-4): ").strip()
    
    if choice == "1" or choice == "4":
        # YOLOv8n Face
        url = "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.onnx"
        output = models_dir / "yolov8n-face.onnx"
        if not output.exists():
            download_file(url, output)
        else:
            print(f"✅ Model đã tồn tại: {output}")
    
    if choice == "2" or choice == "4":
        # YOLOv8n Person (COCO)
        print("\n⚠️  YOLOv8n Person model cần convert từ PyTorch")
        print("Chạy lệnh sau để convert:")
        print("  pip install ultralytics")
        print("  python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')\"")
        print("  mv yolov8n.onnx models/")
    
    if choice == "3" or choice == "4":
        # YOLOv8s Face
        url = "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8s-face.onnx"
        output = models_dir / "yolov8s-face.onnx"
        if not output.exists():
            download_file(url, output)
        else:
            print(f"✅ Model đã tồn tại: {output}")
    
    print("\n" + "=" * 60)
    print("✅ Hoàn thành!")
    print("=" * 60)
    print("\nCập nhật configs/camera_config.json:")
    print('  "type": "yolo_face"  // hoặc "yolo_person"')


if __name__ == "__main__":
    main()



