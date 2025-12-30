"""
Over-the-Air (OTA) Model Update Service
Automatically checks and downloads new model versions from server
"""

import os
import json
import logging
import hashlib
import requests
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelOTAService:
    """Service for OTA model updates"""
    
    def __init__(
        self,
        model_dir: str = "models",
        registry_url: str = "http://localhost:9000",
        bucket_name: str = "models",
        current_model_path: str = "models/multitask_model.onnx"
    ):
        """
        Initialize OTA Service
        
        Args:
            model_dir: Directory to store models
            registry_url: MinIO or model registry URL
            bucket_name: Bucket/container name
            current_model_path: Path to current model file
        """
        self.model_dir = Path(model_dir)
        self.registry_url = registry_url
        self.bucket_name = bucket_name
        self.current_model_path = Path(current_model_path)
        self.version_file = self.model_dir / "model_version.json"
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load current version
        self.current_version = self._load_version()
    
    def _load_version(self) -> Optional[Dict]:
        """Load current model version info"""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load version info: {e}")
        return None
    
    def _get_model_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of model file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def check_for_updates(self) -> Optional[Dict]:
        """
        Check for new model version on server
        
        Returns:
            Version info if update available, None otherwise
        """
        try:
            # Query model registry API
            # In production, this would be a proper API endpoint
            registry_api = f"{self.registry_url}/api/v1/models/latest"
            
            response = requests.get(
                registry_api,
                timeout=10,
                headers={"Accept": "application/json"}
            )
            
            if response.status_code == 200:
                latest_version = response.json()
                
                # Compare versions
                if self.current_version is None:
                    logger.info("No current version, update available")
                    return latest_version
                
                current_ver = self.current_version.get("version", "0.0.0")
                latest_ver = latest_version.get("version", "0.0.0")
                
                if latest_ver > current_ver:
                    logger.info(f"New version available: {latest_ver} (current: {current_ver})")
                    return latest_version
                else:
                    logger.debug(f"Already on latest version: {latest_ver}")
                    return None
            else:
                logger.warning(f"Failed to check updates: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking for updates: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error checking updates: {e}")
            return None
    
    def download_model(self, version_info: Dict) -> Optional[Path]:
        """
        Download new model version
        
        Args:
            version_info: Version information from registry
            
        Returns:
            Path to downloaded model file, None if failed
        """
        try:
            model_url = version_info.get("download_url")
            model_name = version_info.get("name", "multitask_model.onnx")
            version = version_info.get("version", "latest")
            
            # Download model
            logger.info(f"Downloading model version {version}...")
            response = requests.get(model_url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Save to temporary file first
            temp_path = self.model_dir / f"{model_name}.tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify hash if provided
            if "sha256" in version_info:
                calculated_hash = self._get_model_hash(temp_path)
                if calculated_hash != version_info["sha256"]:
                    logger.error("Model hash mismatch, download corrupted")
                    temp_path.unlink()
                    return None
            
            # Move to final location
            final_path = self.model_dir / model_name
            temp_path.rename(final_path)
            
            logger.info(f"Model downloaded: {final_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return None
    
    def update_model(self, new_model_path: Path, version_info: Dict) -> bool:
        """
        Hot-swap current model with new one
        
        Args:
            new_model_path: Path to new model file
            version_info: Version information
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Backup current model
            if self.current_model_path.exists():
                backup_path = self.model_dir / f"{self.current_model_path.stem}_backup.onnx"
                import shutil
                shutil.copy2(self.current_model_path, backup_path)
                logger.info(f"Backed up current model to {backup_path}")
            
            # Replace current model
            import shutil
            shutil.copy2(new_model_path, self.current_model_path)
            
            # Update version info
            version_data = {
                "version": version_info.get("version"),
                "updated_at": datetime.utcnow().isoformat(),
                "sha256": self._get_model_hash(self.current_model_path),
                "path": str(self.current_model_path)
            }
            
            with open(self.version_file, 'w') as f:
                json.dump(version_data, f, indent=2)
            
            logger.info(f"Model updated successfully to version {version_info.get('version')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            return False
    
    def perform_update(self) -> bool:
        """
        Check and perform OTA update if available
        
        Returns:
            True if update performed, False otherwise
        """
        version_info = self.check_for_updates()
        
        if version_info is None:
            return False
        
        new_model_path = self.download_model(version_info)
        
        if new_model_path is None:
            return False
        
        return self.update_model(new_model_path, version_info)

