"""
Face Restoration Module - GFPGAN/ESPCN
Giai đoạn 1 Tuần 3: Advanced Modules
Khôi phục khuôn mặt mờ trước khi nhận diện
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import onnxruntime as ort


class FaceRestorer:
    """
    Face Restoration using ESPCN (lightweight) or GFPGAN
    Enhance blurry/low-quality faces before recognition
    """
    
    def __init__(self, model_path: Optional[str] = None, method: str = "espcn"):
        """
        Initialize Face Restorer
        
        Args:
            model_path: Path to ONNX model
            method: "espcn" (lightweight) or "gfpgan" (better quality)
        """
        self.method = method
        self.model_path = model_path or f"models/{method}.onnx"
        self.session = None
        self.scale_factor = 2 if method == "espcn" else 4
        
        if Path(self.model_path).exists():
            self._load_model()
        else:
            print(f"⚠️  Face restoration model not found: {self.model_path}")
            print("   Face restoration will be disabled")
    
    def _load_model(self):
        """Load ONNX model"""
        try:
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers
            )
            print(f"⚡ Loaded {self.method.upper()} model: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load face restoration model: {e}")
            self.session = None
    
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image
        
        Args:
            face_image: Face image (BGR)
        
        Returns:
            Preprocessed image
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # HWC to CHW
        face_chw = np.transpose(face_normalized, (2, 0, 1))
        
        # Add batch dimension
        face_batch = np.expand_dims(face_chw, axis=0)
        
        return face_batch
    
    def restore(self, face_image: np.ndarray) -> np.ndarray:
        """
        Restore/enhance face image
        
        Args:
            face_image: Low-quality face image (BGR)
        
        Returns:
            Enhanced face image (BGR)
        """
        if self.session is None:
            # If model not loaded, return original
            return face_image
        
        try:
            # Preprocess
            input_data = self.preprocess(face_image)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: input_data})
            
            # Get output
            enhanced = output[0]
            
            # Post-process
            # CHW to HWC
            enhanced_hwc = np.transpose(enhanced[0], (1, 2, 0))
            
            # Denormalize
            enhanced_uint8 = (np.clip(enhanced_hwc, 0, 1) * 255).astype(np.uint8)
            
            # RGB to BGR
            enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)
            
            return enhanced_bgr
        
        except Exception as e:
            print(f"❌ Face restoration error: {e}")
            return face_image  # Return original on error
    
    def enhance_if_needed(self, face_image: np.ndarray, quality_threshold: float = 0.3) -> np.ndarray:
        """
        Enhance face only if quality is low
        
        Args:
            face_image: Face image (BGR)
            quality_threshold: Quality threshold (0-1)
        
        Returns:
            Enhanced face if needed, original otherwise
        """
        # Simple quality check: variance of Laplacian
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize variance (rough estimate)
        quality = min(laplacian_var / 100.0, 1.0)
        
        if quality < quality_threshold:
            return self.restore(face_image)
        else:
            return face_image


if __name__ == "__main__":
    # Test
    restorer = FaceRestorer(method="espcn")
    test_image = np.zeros((64, 64, 3), dtype=np.uint8)
    enhanced = restorer.restore(test_image)
    print(f"Test: Input shape={test_image.shape}, Output shape={enhanced.shape}")

