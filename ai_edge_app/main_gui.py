"""
Edge AI Application - Modern GUI Version
Beautiful, responsive UI with PyQt6 and optimized performance
"""

import sys
import cv2
import json
import time
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from collections import deque
from queue import Queue, Empty

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFrame, QProgressBar
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
    from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt6 not available. Install with: pip install PyQt6")
    print("Falling back to OpenCV display...")

from src.detectors import RetinaFaceDetector, YOLOFaceDetector, YOLOPersonDetector
from src.trackers import DeepSORTTracker, ByteTracker
from src.classifiers import MultiTaskClassifier
from src.ads_engine import AdsSelector
from src.core.dwell_time import DwellTimeTracker
from src.utils import setup_logger


class VideoProcessingThread(QThread if PYQT_AVAILABLE else threading.Thread):
    """Thread for video processing to prevent UI freezing"""
    
    if PYQT_AVAILABLE:
        frame_ready = pyqtSignal(np.ndarray, dict)
        stats_ready = pyqtSignal(dict)
        error_occurred = pyqtSignal(str)
    
    def __init__(self, config_path: str = "configs/camera_config.json"):
        if PYQT_AVAILABLE:
            super().__init__()
        else:
            threading.Thread.__init__(self)
        
        self.config_path = config_path
        self.config = None
        self.running = False
        
        # Components
        self.detector = None
        self.tracker = None
        self.classifier = None
        self.ads_selector = None
        self.dwell_tracker = None
        self.camera = None
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.processed_tracks = {}
        self.track_attributes = {}
        self.tracks_lock = threading.Lock()
        self.last_cleanup_time = time.time()
        
        # Logger
        self.logger = setup_logger('EdgeAIApp', log_file='logs/edge_app.log')
    
    def initialize(self):
        """Initialize all components"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # Initialize detector
            model_dir = Path("models")
            detector_type = self.config['detection'].get('type', 'retinaface')
            
            if detector_type == 'retinaface' or self.detector is None:
                detector_model = model_dir / "retinaface_mnet.onnx"
                if detector_model.exists():
                    self.detector = RetinaFaceDetector(
                        str(detector_model),
                        confidence_threshold=self.config['detection']['confidence_threshold']
                    )
                else:
                    self.detector = RetinaFaceDetector(
                        confidence_threshold=self.config['detection']['confidence_threshold']
                    )
            
            # Initialize tracker
            use_bytetrack = self.config['tracking'].get('use_bytetrack', False)
            if use_bytetrack:
                self.tracker = ByteTracker(
                    track_thresh=self.config['tracking'].get('track_thresh', 0.5),
                    high_thresh=self.config['tracking'].get('high_thresh', 0.6),
                    match_thresh=self.config['tracking'].get('iou_threshold', 0.3),
                    frame_rate=self.config['camera'].get('fps', 30)
                )
            else:
                self.tracker = DeepSORTTracker(
                    max_age=self.config['tracking']['max_age'],
                    min_hits=self.config['tracking']['min_hits'],
                    iou_threshold=self.config['tracking']['iou_threshold']
                )
            
            # Initialize classifier
            classifier_model = model_dir / "multitask_efficientnet.onnx"
            if classifier_model.exists():
                self.classifier = MultiTaskClassifier(str(classifier_model))
            
            # Initialize ads selector
            ads_rules_path = Path("configs/ads_rules.json")
            if ads_rules_path.exists():
                self.ads_selector = AdsSelector(str(ads_rules_path))
            
            # Initialize dwell tracker
            dwell_threshold = self.config['tracking'].get('dwell_threshold', 3.0)
            self.dwell_tracker = DwellTimeTracker(threshold=dwell_threshold)
            
            # Initialize camera
            cam_config = self.config['camera']
            self.camera = cv2.VideoCapture(cam_config['source'])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config['height'])
            
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera: {cam_config['source']}")
            
            self.logger.info("Initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            if PYQT_AVAILABLE:
                self.error_occurred.emit(str(e))
            return False
    
    def run(self):
        """Main processing loop"""
        if not self.camera:
            return
        
        self.running = True
        
        # Performance optimization
        frame_skip = 3  # Process every 3rd frame
        frame_count = 0
        target_fps = 12
        frame_delay = 1.0 / target_fps
        classification_interval = 3.0
        last_classification_time = {}
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                if frame_count % (frame_skip + 1) != 0:
                    time.sleep(frame_delay)
                    continue
                
                start_time = time.time()
                
                # Detection
                detections = []
                if self.detector:
                    try:
                        raw_detections = self.detector.detect(frame)
                        for det in raw_detections:
                            if isinstance(det, tuple) and len(det) >= 5:
                                x, y, w_or_x2, h_or_y2, score = det[0], det[1], det[2], det[3], det[4]
                                if w_or_x2 > 100 or h_or_y2 > 100:
                                    bbox = np.array([x, y, x + w_or_x2, y + h_or_y2], dtype=np.float32)
                                else:
                                    bbox = np.array([x, y, w_or_x2, h_or_y2], dtype=np.float32)
                                detections.append({
                                    'bbox': bbox,
                                    'score': float(score),
                                    'class': 0
                                })
                            elif isinstance(det, dict):
                                detections.append(det)
                    except Exception as e:
                        self.logger.warning(f"Detection error: {e}")
                
                # Tracking
                tracker_result = self.tracker.update(detections)
                
                # Normalize tracker output
                active_tracks = {}
                if isinstance(tracker_result, list):
                    for track in tracker_result:
                        track_id = track['track_id']
                        bbox_array = track['bbox']
                        if len(bbox_array) == 4:
                            if isinstance(bbox_array, np.ndarray):
                                if bbox_array[2] > bbox_array[0] and bbox_array[3] > bbox_array[1]:
                                    x, y, w, h = int(bbox_array[0]), int(bbox_array[1]), \
                                                 int(bbox_array[2] - bbox_array[0]), int(bbox_array[3] - bbox_array[1])
                                else:
                                    x, y, w, h = int(bbox_array[0]), int(bbox_array[1]), \
                                                 int(bbox_array[2]), int(bbox_array[3])
                                active_tracks[track_id] = (x, y, w, h)
                else:
                    active_tracks = tracker_result
                
                # Process tracks
                display_frame = frame.copy()
                current_time = time.time()
                
                if self.dwell_tracker:
                    for track_id in active_tracks.keys():
                        self.dwell_tracker.update_track(track_id, current_time)
                
                # Cleanup old tracks
                if current_time - self.last_cleanup_time > 2.0:
                    self._cleanup_old_tracks(current_time)
                    self.last_cleanup_time = current_time
                
                # Process each track
                tracks_data = []
                for track_id, bbox in active_tracks.items():
                    x, y, w, h = bbox
                    x = max(0, min(x, frame.shape[1] - 1))
                    y = max(0, min(y, frame.shape[0] - 1))
                    w = max(1, min(w, frame.shape[1] - x))
                    h = max(1, min(h, frame.shape[0] - y))
                    
                    # Check if should skip
                    should_skip = False
                    with self.tracks_lock:
                        if track_id in self.processed_tracks:
                            last_time = self.processed_tracks[track_id].get('last_processed', 0)
                            if current_time - last_time < classification_interval:
                                if track_id in self.track_attributes:
                                    attributes = self.track_attributes[track_id].copy()
                                    ad = self.processed_tracks[track_id].get('ad')
                                    tracks_data.append({
                                        'track_id': track_id,
                                        'bbox': bbox,
                                        'attributes': attributes,
                                        'ad': ad
                                    })
                                    should_skip = True
                    
                    if should_skip:
                        continue
                    
                    last_classification_time[track_id] = current_time
                    
                    # Crop face
                    padding = 5
                    x_pad = max(0, x - padding)
                    y_pad = max(0, y - padding)
                    w_pad = min(frame.shape[1] - x_pad, w + 2 * padding)
                    h_pad = min(frame.shape[0] - y_pad, h + 2 * padding)
                    face_roi = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                    
                    # Dwell time check
                    if self.dwell_tracker and not self.dwell_tracker.is_valid_customer(track_id):
                        continue
                    
                    # Classification
                    if (face_roi.size > 0 and len(face_roi.shape) == 3 and
                        face_roi.shape[0] >= 32 and face_roi.shape[1] >= 32 and self.classifier):
                        try:
                            attributes = self.classifier.predict(face_roi)
                            
                            # Ad selection
                            ad = None
                            if self.ads_selector:
                                try:
                                    ad = self.ads_selector.select_ad(
                                        age=attributes['age'],
                                        gender=attributes['gender'],
                                        emotion=attributes['emotion'],
                                        track_id=track_id
                                    )
                                except Exception as e:
                                    self.logger.warning(f"Ad selection error: {e}")
                            
                            with self.tracks_lock:
                                self.track_attributes[track_id] = attributes.copy()
                                old_data = self.processed_tracks.get(track_id, {})
                                start_t = old_data.get('start_time', time.time())
                                self.processed_tracks[track_id] = {
                                    'last_processed': time.time(),
                                    'start_time': start_t,
                                    'ad': ad
                                }
                            
                            tracks_data.append({
                                'track_id': track_id,
                                'bbox': bbox,
                                'attributes': attributes,
                                'ad': ad
                            })
                            
                        except Exception as e:
                            self.logger.warning(f"Classification error: {e}")
                
                # Draw on frame
                display_frame = self._draw_tracks(display_frame, tracks_data)
                
                # Calculate FPS
                processing_time = time.time() - start_time
                fps = 1.0 / processing_time if processing_time > 0 else 0
                self.fps_history.append(fps)
                avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
                
                # Draw FPS
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Tracks: {len(active_tracks)}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Emit frame and stats
                if PYQT_AVAILABLE:
                    self.frame_ready.emit(display_frame, {'tracks': tracks_data})
                    self.stats_ready.emit({
                        'fps': avg_fps,
                        'tracks': len(active_tracks),
                        'total_customers': len(self.processed_tracks)
                    })
                
                # Maintain target FPS
                elapsed = time.time() - start_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                    
        except Exception as e:
            self.logger.error(f"Error in processing loop: {e}", exc_info=True)
            if PYQT_AVAILABLE:
                self.error_occurred.emit(str(e))
        finally:
            self.cleanup()
    
    def _draw_tracks(self, frame, tracks_data):
        """Draw tracks on frame"""
        for track in tracks_data:
            x, y, w, h = track['bbox']
            track_id = track['track_id']
            attributes = track.get('attributes', {})
            ad = track.get('ad')
            
            # Draw bounding box
            color = (0, 255, 0) if not ad else (255, 100, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw info
            info_lines = [
                f"ID: {track_id}",
                f"Age: {attributes.get('age', 'N/A')} - {attributes.get('gender', 'N/A')}",
                f"Emotion: {attributes.get('emotion', 'N/A')}"
            ]
            
            if ad:
                info_lines.append(f"AD: {ad.get('name', 'Promotion')}")
            
            text_y = y - 10
            for i, line in enumerate(reversed(info_lines)):
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                curr_y = text_y - i * 25
                cv2.rectangle(frame, (x, curr_y - 20), (x + text_size[0] + 10, curr_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, line, (x + 5, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def _cleanup_old_tracks(self, current_time):
        """Cleanup old tracks"""
        with self.tracks_lock:
            max_age = 5.0
            tracks_to_remove = []
            
            for track_id, track_data in self.processed_tracks.items():
                last_seen = track_data.get('last_processed', 0)
                if current_time - last_seen > max_age:
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                if self.ads_selector:
                    track_data = self.processed_tracks[track_id]
                    start_time = track_data.get('start_time', track_data.get('last_processed'))
                    duration = current_time - start_time
                    if duration > 0:
                        self.ads_selector.update_feedback(track_id, duration)
                
                self.processed_tracks.pop(track_id, None)
                self.track_attributes.pop(track_id, None)
    
    def stop(self):
        """Stop processing"""
        self.running = False
    
    def cleanup(self):
        """Cleanup resources"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()


if PYQT_AVAILABLE:
    class VideoWidget(QLabel):
        """Video display widget"""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumSize(640, 480)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setStyleSheet("""
                QLabel {
                    background-color: #1a1a1a;
                    border-radius: 8px;
                    border: 2px solid #333;
                }
            """)
        
        def set_frame(self, frame: np.ndarray):
            """Update video frame"""
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
    
    
    class StatsWidget(QFrame):
        """Stats display widget"""
        
        def __init__(self, title: str, parent=None):
            super().__init__(parent)
            self.setFrameShape(QFrame.Shape.StyledPanel)
            self.setStyleSheet("""
                QFrame {
                    background-color: #2a2a2a;
                    border-radius: 8px;
                    border: 1px solid #444;
                    padding: 12px;
                }
            """)
            
            layout = QVBoxLayout()
            self.setLayout(layout)
            
            title_label = QLabel(title)
            title_label.setStyleSheet("color: #aaa; font-size: 12px;")
            layout.addWidget(title_label)
            
            self.value_label = QLabel("0")
            self.value_label.setStyleSheet("color: #fff; font-size: 24px; font-weight: bold;")
            layout.addWidget(self.value_label)
        
        def update_value(self, value: str):
            """Update stat value"""
            self.value_label.setText(value)
    
    
    class MainWindow(QMainWindow):
        """Main application window"""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Smart Retail - Edge AI Analytics")
            self.setGeometry(100, 100, 1280, 720)
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1e1e1e;
                }
            """)
            
            # Central widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            # Main layout
            main_layout = QHBoxLayout()
            central_widget.setLayout(main_layout)
            
            # Left: Video
            left_layout = QVBoxLayout()
            
            self.video_widget = VideoWidget()
            left_layout.addWidget(self.video_widget, stretch=1)
            
            # Control buttons
            button_layout = QHBoxLayout()
            self.start_button = QPushButton("Start")
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: #1890ff;
                    color: white;
                    border-radius: 6px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #40a9ff;
                }
            """)
            self.start_button.clicked.connect(self.start_processing)
            
            self.stop_button = QPushButton("Stop")
            self.stop_button.setStyleSheet("""
                QPushButton {
                    background-color: #ff4d4f;
                    color: white;
                    border-radius: 6px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #ff7875;
                }
            """)
            self.stop_button.clicked.connect(self.stop_processing)
            self.stop_button.setEnabled(False)
            
            button_layout.addWidget(self.start_button)
            button_layout.addWidget(self.stop_button)
            button_layout.addStretch()
            
            left_layout.addLayout(button_layout)
            
            # Right: Stats
            right_layout = QVBoxLayout()
            
            self.fps_widget = StatsWidget("FPS")
            self.tracks_widget = StatsWidget("Active Tracks")
            self.customers_widget = StatsWidget("Total Customers")
            
            right_layout.addWidget(self.fps_widget)
            right_layout.addWidget(self.tracks_widget)
            right_layout.addWidget(self.customers_widget)
            right_layout.addStretch()
            
            main_layout.addLayout(left_layout, stretch=3)
            main_layout.addLayout(right_layout, stretch=1)
            
            # Processing thread
            self.processing_thread = None
        
        def start_processing(self):
            """Start video processing"""
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
            self.processing_thread = VideoProcessingThread()
            self.processing_thread.frame_ready.connect(self.update_frame)
            self.processing_thread.stats_ready.connect(self.update_stats)
            self.processing_thread.error_occurred.connect(self.show_error)
            
            if self.processing_thread.initialize():
                self.processing_thread.start()
            else:
                self.show_error("Failed to initialize")
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
        
        def stop_processing(self):
            """Stop video processing"""
            if self.processing_thread:
                self.processing_thread.stop()
                self.processing_thread.wait()
                self.processing_thread = None
            
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        
        def update_frame(self, frame: np.ndarray, overlay_data: dict):
            """Update video frame"""
            self.video_widget.set_frame(frame)
        
        def update_stats(self, stats: dict):
            """Update statistics"""
            self.fps_widget.update_value(f"{stats.get('fps', 0):.1f}")
            self.tracks_widget.update_value(str(stats.get('tracks', 0)))
            self.customers_widget.update_value(str(stats.get('total_customers', 0)))
        
        def show_error(self, error_msg: str):
            """Show error message"""
            print(f"Error: {error_msg}")
        
        def closeEvent(self, event):
            """Handle window close"""
            self.stop_processing()
            event.accept()


def main():
    """Main entry point"""
    if PYQT_AVAILABLE:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    else:
        # Fallback to OpenCV display
        print("PyQt6 not available. Using OpenCV display...")
        from main import EdgeAIApp
        app = EdgeAIApp()
        try:
            app.initialize()
            app.run()
        except Exception as e:
            print(f"Failed to start: {e}")
            import traceback
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    exit(main())
