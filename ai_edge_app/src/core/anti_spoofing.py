"""
Anti-Spoofing Module - MiniFASNet
Giai đoạn 1 Tuần 3: Advanced Modules
Chống gian lận lượt xem bằng cách lọc bỏ khuôn mặt giả (ảnh/video)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import onnxruntime as ort


class MiniFASNet:
    """
    MiniFASNet Anti-Spoofing Model
    Phát hiện khuôn mặt giả (ảnh, video, mask)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize MiniFASNet
        
        Args:
            model_path: Path to ONNX model
        """
        self.model_path = model_path or "models/minifasnet.onnx"
        self.session = None
        self.input_size = (80, 80)  # MiniFASNet input size
        
        if Path(self.model_path).exists():
            self._load_model()
        else:
            print(f"⚠️  MiniFASNet model not found: {self.model_path}")
            print("   Anti-spoofing will be disabled")
    
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
            print(f"⚡ Loaded MiniFASNet: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load MiniFASNet: {e}")
            self.session = None
    
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for MiniFASNet
        
        Args:
            face_image: Face image (BGR)
        
        Returns:
            Preprocessed image (RGB, normalized)
        """
        # Resize to input size
        face_resized = cv2.resize(face_image, self.input_size)
        
        # BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # HWC to CHW
        face_chw = np.transpose(face_normalized, (2, 0, 1))
        
        # Add batch dimension
        face_batch = np.expand_dims(face_chw, axis=0)
        
        return face_batch
    
    def predict(self, face_image: np.ndarray) -> Tuple[bool, float]:
        """
        Predict if face is real or fake
        
        Args:
            face_image: Face image (BGR)
        
        Returns:
            (is_real, confidence): True if real, False if fake
        """
        if self.session is None:
            # If model not loaded, assume all faces are real
            return True, 1.0
        
        try:
            # Preprocess
            input_data = self.preprocess(face_image)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: input_data})
            
            # Get prediction
            # Output shape: (1, 3) for [real, fake, mask]
            scores = output[0][0]
            real_score = scores[0]
            fake_score = scores[1]
            mask_score = scores[2] if len(scores) > 2 else 0.0
            
            # Determine if real
            is_real = real_score > fake_score and real_score > mask_score
            confidence = real_score if is_real else max(fake_score, mask_score)
            
            return is_real, float(confidence)
        
        except Exception as e:
            print(f"❌ Anti-spoofing prediction error: {e}")
            return True, 1.0  # Default to real on error
    
    def is_real_face(self, face_image: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Check if face is real (convenience method)
        
        Args:
            face_image: Face image (BGR)
            threshold: Confidence threshold
        
        Returns:
            True if real face
        """
        is_real, confidence = self.predict(face_image)
        return is_real and confidence >= threshold


if __name__ == "__main__":
    # Test
    model = MiniFASNet()
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    is_real, conf = model.predict(test_image)
    print(f"Test result: is_real={is_real}, confidence={conf:.2f}")

