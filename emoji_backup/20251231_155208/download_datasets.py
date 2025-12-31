"""
Script to download datasets for training
Downloads face datasets from Kaggle using kagglehub
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports if needed
sys.path.append(str(Path(__file__).parent.parent))

try:
    import kagglehub
except ImportError:
    print("Installing kagglehub...")
    os.system("pip install kagglehub")
    import kagglehub


def download_all_age_face_dataset():
    """
    Download All Age Face Dataset from Kaggle
    Dataset: eshachakraborty00/all-age-face-dataset
    """
    print("=" * 60)
    print("Downloading All Age Face Dataset...")
    print("Dataset: eshachakraborty00/all-age-face-dataset")
    print("=" * 60)
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("eshachakraborty00/all-age-face-dataset")
        print(f"\n✅ Download successful!")
        print(f"📁 Path to dataset files: {path}")
        
        # List contents
        if os.path.exists(path):
            print(f"\n📂 Dataset contents:")
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                else:
                    size = os.path.getsize(item_path) / (1024 * 1024)  # MB
                    print(f"  📄 {item} ({size:.2f} MB)")
        
        return path
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Kaggle account")
        print("   2. Kaggle API credentials set up")
        print("   3. Run: pip install kagglehub")
        return None


def download_utkface_dataset():
    """
    Download UTKFace dataset (if available on Kaggle)
    Alternative: Manual download from https://susanqq.github.io/UTKFace/
    """
    print("\n" + "=" * 60)
    print("UTKFace Dataset")
    print("=" * 60)
    print("ℹ️  UTKFace is typically downloaded manually")
    print("   Visit: https://susanqq.github.io/UTKFace/")
    print("   Or search on Kaggle for 'UTKFace' dataset")
    print("=" * 60)


def download_fer2013_dataset():
    """
    Download FER2013 emotion dataset (if available on Kaggle)
    """
    print("\n" + "=" * 60)
    print("FER2013 Dataset")
    print("=" * 60)
    print("ℹ️  FER2013 is available on Kaggle")
    print("   Search for 'FER2013' or 'facial-expression-recognition'")
    print("=" * 60)


def main():
    """Main function to download all datasets"""
    print("\n" + "=" * 60)
    print("📥 Smart Retail Analytics - Dataset Downloader")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"\n📂 Data directory: {data_dir}")
    
    # Download All Age Face Dataset
    dataset_path = download_all_age_face_dataset()
    
    if dataset_path:
        # Optionally create symlink or copy to data directory
        print(f"\n💡 Dataset saved to: {dataset_path}")
        print(f"   You can use this path in your training scripts")
    
    # Show info about other datasets
    download_utkface_dataset()
    download_fer2013_dataset()
    
    print("\n" + "=" * 60)
    print("✅ Dataset download process completed!")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("   1. Verify dataset files are downloaded correctly")
    print("   2. Check dataset structure and format")
    print("   3. Run data preprocessing scripts")
    print("   4. Start training models")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

