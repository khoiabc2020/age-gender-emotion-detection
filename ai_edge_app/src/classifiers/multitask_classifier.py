"""
Multi-task Learning Model for Age, Gender, and Emotion Classification
Using EfficientNet-B0 backbone optimized with ONNX Runtime
"""

import cv2
import numpy as np
from typing import Dict, Tuple
import onnxruntime as ort


class MultiTaskClassifier:
    """
    Multi-task classifier for age estimation, gender classification, and emotion recognition
    """
    
    def __init__(self, model_path: str):
        """
        Initialize multi-task classifier
        
        Args:
            model_path: Path to ONNX model file
        """
        # Optimize providers: prefer GPU if available
        providers = []
        if ort.get_device() == 'GPU' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (224, 224)
        
        # Emotion labels (6 classes after merging Disgust into Angry)
        # Updated to match Ultimate Edition (6 classes)
        self.emotion_labels = ['angry', 'fear', 'neutral', 'happy', 'sad', 'surprise']
    
    def predict(self, face_image: np.ndarray) -> Dict[str, any]:
        """
        Predict age, gender, and emotion from face image
        Optimized: Better error handling and input validation
        
        Args:
            face_image: Cropped face image (BGR format)
            
        Returns:
            Dictionary with 'age', 'gender', 'emotion', and 'confidence' keys
        """
        # Validate input
        if face_image is None or face_image.size == 0:
            return {
                'age': 30,
                'gender': 'unknown',
                'emotion': 'neutral',
                'confidence': 0.0
            }
        
        try:
            # Preprocess image
            processed = self._preprocess(face_image)
            
            # Run inference (Optimized: Use optimized session)
            outputs = self.session.run(None, {self.input_name: processed})
            
            # Post-process outputs
            predictions = self._post_process(outputs)
            
            return predictions
        except Exception as e:
            # Return default values on error
            return {
                'age': 30,
                'gender': 'unknown',
                'emotion': 'neutral',
                'confidence': 0.0
            }
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: Input face image
            
        Returns:
            Preprocessed image tensor
        """
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Normalize to [0, 1] and convert to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Apply normalization (ImageNet mean/std)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Convert to CHW format and add batch dimension
        chw = np.transpose(normalized, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)
        
        return batch.astype(np.float32)
    
    def _post_process(self, outputs: list) -> Dict[str, any]:
        """
        Post-process model outputs
        
        Args:
            outputs: Model output tensors [age, gender_logits, emotion_logits]
            
        Returns:
            Dictionary with predictions
        """
        try:
            # Validate outputs
            if len(outputs) < 3:
                raise ValueError(f"Expected 3 outputs, got {len(outputs)}")
            
            # Extract outputs (handle batch dimension)
            age_output = outputs[0]
            gender_output = outputs[1]
            emotion_output = outputs[2]
            
            # Remove batch dimension if present
            if len(age_output.shape) > 1:
                age_output = age_output[0]
            if len(gender_output.shape) > 1:
                gender_output = gender_output[0]
            if len(emotion_output.shape) > 1:
                emotion_output = emotion_output[0]
            
            # Age (regression) - output is scalar or 1D array
            if age_output.ndim > 0:
                age = float(age_output[0] if len(age_output) > 0 else age_output)
            else:
                age = float(age_output)
            age = max(0, min(100, age))  # Clamp to valid range
            
            # Gender (binary classification)
            # Model outputs: [male_logit, female_logit] based on training
            gender_probs = self._softmax(gender_output)
            if len(gender_probs) >= 2:
                # Index 0 = male, Index 1 = female (based on training dataset format)
                gender = 'male' if gender_probs[0] > gender_probs[1] else 'female'
                gender_confidence = float(max(gender_probs))
            else:
                # Fallback for unexpected output shape
                gender = 'male' if gender_probs[0] > 0.5 else 'female'
                gender_confidence = float(max(gender_probs[0], 1 - gender_probs[0]))
            
            # Emotion (multi-class classification)
            emotion_probs = self._softmax(emotion_output)
            if len(emotion_probs) != len(self.emotion_labels):
                raise ValueError(f"Emotion output size {len(emotion_probs)} doesn't match labels {len(self.emotion_labels)}")
            
            emotion_idx = int(np.argmax(emotion_probs))
            emotion = self.emotion_labels[emotion_idx]
            emotion_confidence = float(emotion_probs[emotion_idx])
            
            return {
                'age': int(round(age)),
                'gender': gender,
                'emotion': emotion,
                'confidence': {
                    'gender': gender_confidence,
                    'emotion': emotion_confidence
                }
            }
        except Exception as e:
            # Return default values on error
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in post-processing: {e}")
            return {
                'age': 30,
                'gender': 'male',
                'emotion': 'neutral',
                'confidence': {
                    'gender': 0.5,
                    'emotion': 0.5
                }
            }
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

