"""
Script to check and summarize downloaded datasets
"""

import os
from pathlib import Path
import kagglehub


def get_dataset_path(dataset_id):
    """Get cached dataset path"""
    try:
        path = kagglehub.dataset_download(dataset_id)
        return path
    except:
        return None


def count_files(directory):
    """Count files in directory recursively"""
    count = 0
    total_size = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
            except:
                pass
    return count, total_size / (1024 * 1024)  # Return count and size in MB


def check_dataset_structure(path, dataset_name):
    """Check and summarize dataset structure"""
    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)
    
    if not path or not os.path.exists(path):
        print(f"[ERROR] Path not found: {path}")
        return None
    
    print(f"Location: {path}")
    
    # Count files and size
    file_count, total_size = count_files(path)
    print(f"Total files: {file_count:,}")
    print(f"Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    # List top-level structure
    print("\nTop-level structure:")
    items = os.listdir(path)
    for item in items[:10]:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            sub_count, sub_size = count_files(item_path)
            print(f"  [DIR]  {item}/")
            print(f"         -> {sub_count:,} files, {sub_size:.2f} MB")
        else:
            size = os.path.getsize(item_path) / (1024 * 1024)
            print(f"  [FILE] {item} ({size:.2f} MB)")
    
    if len(items) > 10:
        print(f"  ... and {len(items) - 10} more items")
    
    return {
        "path": path,
        "file_count": file_count,
        "total_size_mb": total_size
    }


def main():
    """Check all datasets"""
    print("\n" + "=" * 60)
    print("Dataset Checker - Smart Retail Analytics")
    print("=" * 60)
    
    datasets = {
        "All Age Face Dataset": "eshachakraborty00/all-age-face-dataset",
        "UTKFace Dataset": "jangedoo/utkface-new",
        "FER2013 Dataset": "msambare/fer2013"
    }
    
    results = {}
    
    for name, dataset_id in datasets.items():
        path = get_dataset_path(dataset_id)
        result = check_dataset_structure(path, name)
        if result:
            results[name] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Datasets checked: {len(results)}/{len(datasets)}")
    
    total_files = sum(r["file_count"] for r in results.values())
    total_size = sum(r["total_size_mb"] for r in results.values())
    
    print(f"\nTotal files across all datasets: {total_files:,}")
    print(f"Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    print("\n" + "=" * 60)
    print("Datasets are ready for preprocessing and training!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

