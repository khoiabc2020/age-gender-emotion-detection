"""
RetinaFace Face Detector
Optimized for edge computing with ONNX Runtime
Tuần 5: Face Detection & Tracking pipeline
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import onnxruntime as ort
from pathlib import Path


class RetinaFaceDetector:
    """
    Face detection using RetinaFace model optimized with ONNX Runtime
    
    Supports both ONNX model and fallback to OpenCV DNN if model not available
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.7):
        """
        Initialize RetinaFace detector
        
        Args:
            model_path: Path to ONNX model file (optional, will use OpenCV DNN if not provided)
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.input_size = (640, 640)  # Standard input size for RetinaFace
        
        # Use OpenCV DNN as fallback
        self.use_opencv_dnn = False
        self.net = None
        
        # Initialize logger first
        self.logger = self._get_logger()
        
        if model_path and Path(model_path).exists():
            try:
                # Try to load ONNX model with optimized providers
                # Prefer GPU if available, fallback to CPU
                providers = []
                if ort.get_device() == 'GPU' and 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.append('CUDAExecutionProvider')
                providers.append('CPUExecutionProvider')
                
                self.session = ort.InferenceSession(
                    model_path,
                    providers=providers
                )
                self.input_name = self.session.get_inputs()[0].name
                self.logger.info(f"Loaded RetinaFace ONNX model from {model_path} (Providers: {providers})")
            except (RuntimeError, ValueError, FileNotFoundError) as e:
                self.logger.warning(f"Failed to load ONNX model: {e}. Using OpenCV DNN fallback.")
                self.session = None
                self._init_opencv_dnn()
        else:
            # Use OpenCV DNN as fallback
            self._init_opencv_dnn()
    
    def _init_opencv_dnn(self):
        """Initialize OpenCV DNN face detector as fallback"""
        if not hasattr(self, 'logger'):
            self.logger = self._get_logger()
        
        try:
            # Try to load OpenCV DNN face detector
            prototxt_path = Path(__file__).parent.parent.parent / "models" / "deploy.prototxt"
            caffemodel_path = Path(__file__).parent.parent.parent / "models" / "res10_300x300_ssd_iter_140000.caffemodel"
            
            if prototxt_path.exists() and caffemodel_path.exists():
                self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
                self.use_opencv_dnn = True
                self.input_size = (300, 300)
                self.logger.info("Using OpenCV DNN face detector as fallback")
            else:
                # Use Haar Cascade as last resort
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if not Path(cascade_path).exists():
                    # Try alternative path
                    cascade_path = str(Path(cv2.__file__).parent / 'data' / 'haarcascade_frontalface_default.xml')
                
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")
                
                self.use_opencv_dnn = False
                self.logger.info("Using Haar Cascade face detector")
        except (FileNotFoundError, RuntimeError, cv2.error) as e:
            self.logger.error(f"Failed to initialize face detector: {e}")
            # Set to None to prevent crashes
            self.net = None
            self.face_cascade = None
            self.use_opencv_dnn = False
    
    def _get_logger(self):
        """Get logger instance"""
        try:
            from src.utils.logger import setup_logger
            return setup_logger('RetinaFaceDetector')
        except (ImportError, AttributeError) as e:
            import logging
            logger = logging.getLogger('RetinaFaceDetector')
            logger.warning(f"Could not setup custom logger: {e}, using default logger")
            return logger
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of (x, y, w, h, confidence) bounding boxes
        """
        if self.session:
            return self._detect_onnx(image)
        elif self.use_opencv_dnn and self.net:
            return self._detect_opencv_dnn(image)
        else:
            return self._detect_haar(image)
    
    def _detect_onnx(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using ONNX model"""
        h, w = image.shape[:2]
        
        # Preprocess image
        # RetinaFace ONNX models may have different input formats
        # Use blobFromImage for consistency (same as OpenCV DNN)
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Create blob (handles normalization and format conversion)
        input_blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0,
            size=self.input_size,
            mean=[104, 117, 123],  # BGR mean for ImageNet
            swapRB=False,  # Keep BGR (OpenCV default)
            crop=False
        )
        
        # Convert to numpy array if needed
        input_array = input_blob.astype(np.float32)
        
        # Run inference
        try:
            outputs = self.session.run(None, {self.input_name: input_array})
        except Exception as e:
            self.logger.error(f"ONNX inference failed: {e}")
            return []
        
        # Post-process detections
        detections = self._post_process_onnx(outputs, (h, w))
        
        return detections
    
    def _detect_opencv_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using OpenCV DNN"""
        h, w = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, self.input_size),
            1.0,
            self.input_size,
            [104, 117, 123]
        )
        
        # Set input and forward pass
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Post-process
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)  # Fixed: use h instead of w
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)  # Fixed: use h instead of w
                
                # Convert to (x, y, w, h) format
                x = max(0, x1)
                y = max(0, y1)
                w_box = max(1, x2 - x1)
                h_box = max(1, y2 - y1)
                
                faces.append((x, y, w_box, h_box, float(confidence)))
        
        return faces
    
    def _detect_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using Haar Cascade (fallback)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to (x, y, w, h, confidence) format
        detections = []
        for (x, y, w, h) in faces:
            detections.append((x, y, w, h, 0.8))  # Default confidence for Haar
        
        return detections
    
    def _post_process_onnx(
        self, 
        outputs: List[np.ndarray], 
        original_shape: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Post-process ONNX model outputs to get bounding boxes
        
        Args:
            outputs: Model output tensors
            original_shape: Original image shape (H, W)
            
        Returns:
            List of bounding boxes (x, y, w, h, confidence)
        """
        detections = []
        h, w = original_shape
        scale_x = w / self.input_size[0]
        scale_y = h / self.input_size[1]
        
        # RetinaFace typically outputs: [boxes, scores, landmarks]
        # Format may vary, so we handle common formats
        if len(outputs) >= 2:
            boxes = outputs[0]  # Shape: (N, 4) or (1, N, 4)
            scores = outputs[1]  # Shape: (N,) or (1, N)
            
            # Handle batch dimension
            if len(boxes.shape) == 3:
                boxes = boxes[0]
            if len(scores.shape) == 2:
                scores = scores[0]
            
            # Process each detection
            for i in range(len(scores)):
                if scores[i] > self.confidence_threshold:
                    # Get bounding box coordinates
                    if boxes.shape[1] == 4:
                        # Format: [x1, y1, x2, y2] or [x, y, w, h]
                        box = boxes[i]
                        
                        # Check if format is [x1, y1, x2, y2] or [x, y, w, h]
                        if box[2] > box[0] and box[3] > box[1]:
                            # Format: [x1, y1, x2, y2]
                            x1 = int(box[0] * scale_x)
                            y1 = int(box[1] * scale_y)
                            x2 = int(box[2] * scale_x)
                            y2 = int(box[3] * scale_y)
                            
                            x = max(0, x1)
                            y = max(0, y1)
                            w_box = max(1, x2 - x1)
                            h_box = max(1, y2 - y1)
                        else:
                            # Format: [x, y, w, h]
                            x = int(box[0] * scale_x)
                            y = int(box[1] * scale_y)
                            w_box = int(box[2] * scale_x)
                            h_box = int(box[3] * scale_y)
                        
                        # Ensure valid bounding box
                        x = max(0, min(x, w - 1))
                        y = max(0, min(y, h - 1))
                        w_box = max(1, min(w_box, w - x))
                        h_box = max(1, min(h_box, h - y))
                        
                        detections.append((x, y, w_box, h_box, float(scores[i])))
        
        return detections
