"""
Script to copy datasets from kagglehub cache to project directory
"""
import os
import shutil
from pathlib import Path

def get_folder_size(filepath):
    """Calculate total size of folder in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(filepath):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # MB

def copy_dataset(source, dest, dataset_name):
    """Copy dataset from source to destination"""
    print(f"\n{'='*60}")
    print(f"Copying {dataset_name}...")
    print(f"{'='*60}")
    
    if not os.path.exists(source):
        print(f"[ERROR] Source path does not exist: {source}")
        return False
    
    # Create destination directory
    os.makedirs(dest, exist_ok=True)
    
    # Get source size
    source_size = get_folder_size(source)
    print(f"Source: {source}")
    print(f"Destination: {dest}")
    print(f"Size: {source_size:.2f} MB")
    print(f"Copying files...")
    
    try:
        # Copy entire directory tree
        if os.path.isdir(source):
            # Copy all contents
            for item in os.listdir(source):
                source_item = os.path.join(source, item)
                dest_item = os.path.join(dest, item)
                
                if os.path.isdir(source_item):
                    if os.path.exists(dest_item):
                        shutil.rmtree(dest_item)
                    shutil.copytree(source_item, dest_item)
                    print(f"  [DIR] Copied {item}/")
                else:
                    shutil.copy2(source_item, dest_item)
                    print(f"  [FILE] Copied {item}")
        
        dest_size = get_folder_size(dest)
        print(f"\n[SUCCESS] {dataset_name} copied successfully!")
        print(f"Final size: {dest_size:.2f} MB")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to copy {dataset_name}: {e}")
        return False

def main():
    # Project root directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # Kagglehub cache paths
    cache_base = Path.home() / ".cache" / "kagglehub" / "datasets"
    
    datasets = [
        {
            "name": "All Age Face Dataset",
            "source": cache_base / "eshachakraborty00" / "all-age-face-dataset" / "versions" / "1",
            "dest": data_dir / "all_age_face_dataset"
        },
        {
            "name": "UTKFace Dataset",
            "source": cache_base / "jangedoo" / "utkface-new" / "versions" / "1",
            "dest": data_dir / "utkface"
        },
        {
            "name": "FER2013 Dataset",
            "source": cache_base / "msambare" / "fer2013" / "versions" / "1",
            "dest": data_dir / "fer2013"
        }
    ]
    
    print("\n" + "="*60)
    print("Copying Datasets to Project Directory")
    print("="*60)
    print(f"\nProject data directory: {data_dir}")
    print(f"Kagglehub cache: {cache_base}")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    success_count = 0
    for dataset in datasets:
        if copy_dataset(str(dataset["source"]), str(dataset["dest"]), dataset["name"]):
            success_count += 1
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Successfully copied: {success_count}/{len(datasets)} datasets")
    print(f"\nDatasets location in project:")
    for dataset in datasets:
        if os.path.exists(dataset["dest"]):
            size = get_folder_size(dataset["dest"])
            print(f"  - {dataset['name']}: {dataset['dest']} ({size:.2f} MB)")
    
    if success_count == len(datasets):
        print("\n[SUCCESS] All datasets copied to project directory!")
        print("You can now use them from: training_experiments/data/")
    else:
        print("\n[WARNING] Some datasets failed to copy. Please check errors above.")

if __name__ == "__main__":
    main()

