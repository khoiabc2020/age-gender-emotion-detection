"""
Face/Person Detection Modules
Supports multiple detection backends: RetinaFace, YOLO
"""

from .retinaface_detector import RetinaFaceDetector
from .yolo_detector import YOLODetector, YOLOFaceDetector, YOLOPersonDetector

__all__ = [
    'RetinaFaceDetector',
    'YOLODetector',
    'YOLOFaceDetector',
    'YOLOPersonDetector'
]
