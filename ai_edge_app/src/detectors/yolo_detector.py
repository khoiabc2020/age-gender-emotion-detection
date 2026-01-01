"""
YOLO Face/Person Detector
Optimized for edge computing with ONNX Runtime
Hỗ trợ YOLOv5, YOLOv8 cho face detection hoặc person detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import onnxruntime as ort
from pathlib import Path

class YOLODetector:
    """
    YOLO detector for face or person detection
    Supports YOLOv5/YOLOv8 models in ONNX format
    
    Advantages over RetinaFace:
    - Faster inference (especially YOLOv8)
    - Can detect full body (person detection)
    - Better for multiple objects
    - More flexible (can detect other objects too)
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        detect_class: str = "face"  # "face" or "person"
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to ONNX model file
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
            input_size: Model input size (width, height)
            detect_class: "face" or "person" (depends on model training)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.detect_class = detect_class
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_names = None
        
        # Initialize logger
        self.logger = self._get_logger()
        
        if model_path and Path(model_path).exists():
            try:
                # Optimize providers: prefer GPU if available
                providers = []
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.append('CUDAExecutionProvider')
                providers.append('CPUExecutionProvider')
                
                self.session = ort.InferenceSession(
                    model_path,
                    providers=providers
                )
                
                # Get input/output names
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [output.name for output in self.session.get_outputs()]
                
                # Get input shape to determine input size if not specified
                input_shape = self.session.get_inputs()[0].shape
                if len(input_shape) >= 3:
                    # Format: (batch, channels, height, width)
                    if input_shape[2] == input_shape[3]:
                        self.input_size = (int(input_shape[3]), int(input_shape[2]))
                
                self.logger.info(f"Loaded YOLO ONNX model from {model_path}")
                self.logger.info(f"Input: {self.input_name}, Outputs: {self.output_names}")
                self.logger.info(f"Input size: {self.input_size}, Providers: {providers}")
            except Exception as e:
                self.logger.error(f"Failed to load YOLO model: {e}")
                self.session = None
        else:
            self.logger.warning(f"YOLO model not found: {model_path}")
            self.logger.warning("Please download YOLO model or use RetinaFace detector")
    
    def _get_logger(self):
        """Get logger instance"""
        try:
            from src.utils.logger import setup_logger
            return setup_logger('YOLODetector')
        except Exception:
            import logging
            logger = logging.getLogger('YOLODetector')
            logger.warning("Could not setup custom logger, using default logger")
            return logger
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces or persons in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of (x, y, w, h, confidence) bounding boxes
        """
        if not self.session:
            return []
        
        try:
            return self._detect_yolo(image)
        except Exception as e:
            self.logger.error(f"YOLO detection error: {e}")
            return []
    
    def _detect_yolo(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect using YOLO ONNX model"""
        h, w = image.shape[:2]
        
        # Preprocess image
        input_tensor = self._preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Post-process outputs
        detections = self._post_process(outputs, (h, w))
        
        return detections
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO model
        
        YOLO expects:
        - RGB format
        - Normalized to [0, 1]
        - Resized to input_size (with letterbox to maintain aspect ratio)
        - CHW format with batch dimension
        """
        # Resize with letterbox (maintain aspect ratio)
        resized, ratio, pad = self._letterbox(image, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert to CHW format
        chw = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batch = np.expand_dims(chw, axis=0)
        
        return batch.astype(np.float32)
    
    def _letterbox(
        self, 
        image: np.ndarray, 
        new_shape: Tuple[int, int],
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize image with letterbox padding (maintain aspect ratio)
        Returns: resized_image, ratio, (pad_w, pad_h)
        """
        shape = image.shape[:2]  # Current shape [height, width]
        
        # Calculate ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Calculate new size
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        
        # Resize
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        # Add padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return image, r, (left, top)
    
    def _post_process(
        self, 
        outputs: List[np.ndarray], 
        original_shape: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Post-process YOLO outputs
        
        YOLO output format varies:
        - YOLOv5: (1, num_detections, 6) = [x, y, w, h, conf, class]
        - YOLOv8: (1, 84, 8400) or (1, num_detections, 84) = [x, y, w, h, 80 classes]
        - YOLOv8-face: (1, num_detections, 6) = [x, y, w, h, conf, class]
        """
        detections = []
        h, w = original_shape
        
        if len(outputs) == 0:
            return detections
        
        # Get first output
        predictions = outputs[0]  # Shape varies
        
        # Handle different YOLO output formats
        # YOLOv8 format: (1, 84, 8400) - need to transpose and reshape
        if len(predictions.shape) == 3 and predictions.shape[0] == 1:
            # Check if it's YOLOv8 format (84 = 4 box + 80 classes)
            if predictions.shape[1] == 84 or (predictions.shape[1] > 80 and predictions.shape[1] < 100):
                # YOLOv8 format: (1, 84, num_anchors)
                # Transpose to (1, num_anchors, 84)
                predictions = np.transpose(predictions, (0, 2, 1))
        
        # Remove batch dimension
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # (num_detections, features)
        
        # Process each detection
        boxes = []
        confidences = []
        
        for pred in predictions:
            # YOLO format: [x_center, y_center, width, height, confidence, ...classes]
            # Coordinates are normalized [0, 1] relative to input_size
            
            if len(pred) < 4:
                continue
            
            # Get box coordinates (normalized [0, 1])
            x_center = float(pred[0])
            y_center = float(pred[1])
            box_w = float(pred[2])
            box_h = float(pred[3])
            
            # Validate coordinates
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                    0 < box_w <= 1 and 0 < box_h <= 1):
                continue
            
            # Get confidence and class
            confidence = 1.0
            class_id = 0
            
            if len(pred) >= 5:
                # YOLOv5/YOLOv8-face format: [x, y, w, h, conf, class]
                if len(pred) == 6:
                    confidence = float(pred[4])
                    class_id = int(pred[5])
                # YOLOv8 COCO format: [x, y, w, h, class_scores...]
                elif len(pred) > 5:
                    # Get class scores (80 classes for COCO)
                    class_scores = pred[4:]
                    class_id = int(np.argmax(class_scores))
                    confidence = float(class_scores[class_id])
            
            # Filter by confidence
            if confidence < self.confidence_threshold:
                continue
            
            # Filter by class if needed
            # For face detection: class_id should be 0 (face)
            # For person detection: class_id should be 0 (person) in COCO
            if self.detect_class == "face" and class_id != 0:
                continue
            elif self.detect_class == "person" and class_id != 0:
                continue
            
            # Convert from center format to corner format
            # Coordinates are normalized [0, 1], scale to input_size
            x1 = (x_center - box_w / 2) * self.input_size[0]
            y1 = (y_center - box_h / 2) * self.input_size[1]
            x2 = (x_center + box_w / 2) * self.input_size[0]
            y2 = (y_center + box_h / 2) * self.input_size[1]
            
            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))
        
        # Apply Non-Maximum Suppression (NMS)
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            confidences = np.array(confidences, dtype=np.float32)
            
            # OpenCV NMS
            try:
                indices = cv2.dnn.NMSBoxes(
                    boxes.tolist(),
                    confidences.tolist(),
                    self.confidence_threshold,
                    self.iou_threshold
                )
            except Exception as e:
                self.logger.warning(f"NMS error: {e}, returning all detections")
                indices = np.arange(len(boxes))
            
            if len(indices) > 0:
                # Flatten indices (OpenCV returns array of arrays)
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                
                # Scale boxes to original image size
                # Note: YOLO outputs are relative to input_size (with letterbox padding)
                # For simplicity, we scale directly (letterbox padding is usually small)
                scale_x = w / self.input_size[0]
                scale_y = h / self.input_size[1]
                
                for idx in indices:
                    box = boxes[idx]
                    conf = confidences[idx]
                    
                    # Scale to original image
                    x1 = int(box[0] * scale_x)
                    y1 = int(box[1] * scale_y)
                    x2 = int(box[2] * scale_x)
                    y2 = int(box[3] * scale_y)
                    
                    # Convert to (x, y, w, h) format
                    x = max(0, x1)
                    y = max(0, y1)
                    w_box = max(1, x2 - x1)
                    h_box = max(1, y2 - y1)
                    
                    # Ensure within image bounds
                    x = min(x, w - 1)
                    y = min(y, h - 1)
                    w_box = min(w_box, w - x)
                    h_box = min(h_box, h - y)
                    
                    detections.append((x, y, w_box, h_box, float(conf)))
        
        return detections

class YOLOFaceDetector(YOLODetector):
    """
    Specialized YOLO detector for face detection
    Uses YOLOv5-face or YOLOv8-face models
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            input_size=(640, 640),
            detect_class="face"
        )

class YOLOPersonDetector(YOLODetector):
    """
    Specialized YOLO detector for person detection
    Uses YOLOv5 or YOLOv8 COCO models
    Can detect full body, useful for tracking entire person
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            input_size=(640, 640),
            detect_class="person"
        )
    
    def detect_persons(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons (full body) in image
        Returns bounding boxes for entire person, not just face
        """
        return self.detect(image)
