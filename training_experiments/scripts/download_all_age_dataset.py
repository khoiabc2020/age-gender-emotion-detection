"""
Script to download multiple datasets for training
Downloads: All Age Face, UTKFace, and FER2013 datasets
"""

import kagglehub
import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def download_dataset(dataset_name, dataset_id, description):
    """Download a dataset from Kaggle"""
    print("\n" + "=" * 60)
    print(f"Downloading {dataset_name}...")
    print(f"Dataset: {dataset_id}")
    print(f"Description: {description}")
    print("=" * 60)
    
    try:
        path = kagglehub.dataset_download(dataset_id)
        print(f"\n[SUCCESS] Download completed!")
        print(f"Path: {path}")
        
        # List contents
        if os.path.exists(path):
            print("\nDataset contents:")
            items = os.listdir(path)
            for item in items[:10]:  # Show first 10 items
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    subitems = len(os.listdir(item_path))
                    print(f"  [DIR]  {item}/ ({subitems} items)")
                else:
                    size = os.path.getsize(item_path) / (1024 * 1024)  # MB
                    print(f"  [FILE] {item} ({size:.2f} MB)")
            if len(items) > 10:
                print(f"  ... and {len(items) - 10} more items")
        
        return path
    except Exception as e:
        print(f"\n[ERROR] Failed to download {dataset_name}: {e}")
        return None


def main():
    """Download all datasets"""
    print("\n" + "=" * 60)
    print("Smart Retail Analytics - Dataset Downloader")
    print("=" * 60)
    
    datasets = [
        {
            "name": "All Age Face Dataset",
            "id": "eshachakraborty00/all-age-face-dataset",
            "description": "Face images with age labels for age estimation"
        },
        {
            "name": "UTKFace Dataset",
            "id": "jangedoo/utkface-new",
            "description": "Large-scale face dataset with age, gender, and ethnicity labels"
        },
        {
            "name": "FER2013 Dataset",
            "id": "msambare/fer2013",
            "description": "Facial Expression Recognition dataset with 7 emotion classes"
        }
    ]
    
    downloaded_paths = {}
    
    for dataset in datasets:
        path = download_dataset(
            dataset["name"],
            dataset["id"],
            dataset["description"]
        )
        if path:
            downloaded_paths[dataset["name"]] = path
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Successfully downloaded: {len(downloaded_paths)}/{len(datasets)} datasets")
    
    for name, path in downloaded_paths.items():
        print(f"\n{name}:")
        print(f"  Location: {path}")
    
    print("\n" + "=" * 60)
    print("All datasets ready for training!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check dataset structures")
    print("2. Run preprocessing scripts")
    print("3. Start model training")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

