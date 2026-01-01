"""
Multi-Threading Architecture với QThread
System Logic & Optimization
QThread: Grabber, Inferencer, Renderer với Queue-based pipeline
"""

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from queue import Queue, Empty
import cv2
import numpy as np
import time
from typing import Optional, Dict, Any

class FrameGrabber(QThread):
    """
    Thread 1: Grabber
    Chỉ đọc camera, đẩy frame vào Queue (tốc độ cao)
    """
    
    frame_ready = pyqtSignal(np.ndarray, float)  # frame, timestamp
    
    def __init__(self, camera_source: int = 0, frame_queue: Queue = None):
        super().__init__()
        self.camera_source = camera_source
        self.frame_queue = frame_queue or Queue(maxsize=2)
        self.running = False
        self.camera = None
    
    def run(self):
        """Main grabber loop"""
        self.camera = cv2.VideoCapture(self.camera_source)
        if not self.camera.isOpened():
            print(f"ERROR: Cannot open camera {self.camera_source}")
            return
        
        self.running = True
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            timestamp = time.time()
            
            # Put frame in queue (non-blocking)
            try:
                self.frame_queue.put_nowait((frame.copy(), timestamp))
            except:
                # Queue full, skip frame
                try:
                    self.frame_queue.get_nowait()  # Remove oldest
                    self.frame_queue.put_nowait((frame.copy(), timestamp))
                except:
                    pass
            
            # Emit signal for direct connection
            self.frame_ready.emit(frame, timestamp)
    
    def stop(self):
        """Stop grabber"""
        self.running = False
        if self.camera:
            self.camera.release()

class FrameInferencer(QThread):
    """
    Thread 2: Inferencer
    Lấy frame từ Queue, chạy AI, đẩy kết quả vào ResultQueue
    """
    
    result_ready = pyqtSignal(dict)  # result dict
    
    def __init__(
        self,
        frame_queue: Queue,
        result_queue: Queue,
        detector=None,
        tracker=None,
        classifier=None,
        anti_spoofing=None,
        face_restorer=None,
        dwell_tracker=None
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue or Queue(maxsize=5)
        self.detector = detector
        self.tracker = tracker
        self.classifier = classifier
        self.anti_spoofing = anti_spoofing
        self.face_restorer = face_restorer
        self.dwell_tracker = dwell_tracker
        self.running = False
    
    def run(self):
        """Main inference loop"""
        self.running = True
        
        while self.running:
            try:
                # Get frame from queue (blocking with timeout)
                frame, timestamp = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue
            
            result = {
                'frame': frame,
                'timestamp': timestamp,
                'detections': [],
                'tracks': {},
                'attributes': {}
            }
            
            # Step 1: Detection
            if self.detector:
                detections = self.detector.detect(frame)
                result['detections'] = detections
            
            # Step 2: Tracking
            if self.tracker and detections:
                if hasattr(self.tracker, 'update'):
                    # ByteTrack or DeepSORT
                    tracks = self.tracker.update(detections)
                    result['tracks'] = tracks
                    
                    # Update dwell time
                    if self.dwell_tracker:
                        current_time = time.time()
                        for track_id in tracks.keys() if isinstance(tracks, dict) else [t['track_id'] for t in tracks]:
                            self.dwell_tracker.update_track(track_id, current_time)
            
            # Step 3: Classification (for valid customers only)
            if self.classifier and result['tracks']:
                # Normalize tracks format
                if isinstance(result['tracks'], dict):
                    # DeepSORT format: {track_id: (x, y, w, h)}
                    tracks_list = [(tid, bbox) for tid, bbox in result['tracks'].items()]
                else:
                    # ByteTrack format: [{'track_id', 'bbox', ...}, ...]
                    tracks_list = []
                    for track in result['tracks']:
                        track_id = track.get('track_id', 0)
                        bbox_array = track.get('bbox', [])
                        if len(bbox_array) == 4:
                            if isinstance(bbox_array, np.ndarray):
                                if bbox_array[2] > bbox_array[0] and bbox_array[3] > bbox_array[1]:
                                    # [x1, y1, x2, y2] format
                                    x, y, w, h = int(bbox_array[0]), int(bbox_array[1]), \
                                                 int(bbox_array[2] - bbox_array[0]), int(bbox_array[3] - bbox_array[1])
                                else:
                                    # [x, y, w, h] format
                                    x, y, w, h = int(bbox_array[0]), int(bbox_array[1]), \
                                                 int(bbox_array[2]), int(bbox_array[3])
                                tracks_list.append((track_id, (x, y, w, h)))
                
                for track_id, bbox in tracks_list:
                    # Check dwell time ()
                    if self.dwell_tracker and not self.dwell_tracker.is_valid_customer(track_id):
                        continue
                    
                    # Crop face
                    x, y, w, h = bbox
                    x = max(0, min(x, frame.shape[1] - 1))
                    y = max(0, min(y, frame.shape[0] - 1))
                    w = max(1, min(w, frame.shape[1] - x))
                    h = max(1, min(h, frame.shape[0] - y))
                    
                    face_roi = frame[y:y+h, x:x+w]
                    
                    if face_roi.size > 0 and len(face_roi.shape) == 3 and face_roi.shape[0] >= 32 and face_roi.shape[1] >= 32:
                        # Anti-spoofing ()
                        if self.anti_spoofing:
                            is_real, _ = self.anti_spoofing.predict(face_roi)
                            if not is_real:
                                continue
                        
                        # Face restoration ()
                        if self.face_restorer:
                            face_roi = self.face_restorer.enhance_if_needed(face_roi, quality_threshold=0.3)
                        
                        # Classification
                        attributes = self.classifier.predict(face_roi)
                        result['attributes'][track_id] = attributes
            
            # Put result in queue
            try:
                self.result_queue.put_nowait(result)
            except:
                # Queue full, remove oldest
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.put_nowait(result)
                except:
                    pass
            
            # Emit signal
            self.result_ready.emit(result)
    
    def stop(self):
        """Stop inferencer"""
        self.running = False

class FrameRenderer(QThread):
    """
    Thread 3: Renderer
    Lấy kết quả từ ResultQueue, vẽ UI
    """
    
    frame_rendered = pyqtSignal(np.ndarray)  # rendered frame
    
    def __init__(self, result_queue: Queue, ads_selector=None):
        super().__init__()
        self.result_queue = result_queue
        self.ads_selector = ads_selector
        self.running = False
    
    def run(self):
        """Main renderer loop"""
        self.running = True
        
        while self.running:
            try:
                # Get result from queue
                result = self.result_queue.get(timeout=0.1)
            except Empty:
                continue
            
            frame = result['frame'].copy()
            
            # Draw tracks, attributes, ads
            # This will be enhanced with smart overlay
            
            # Emit rendered frame
            self.frame_rendered.emit(frame)
    
    def stop(self):
        """Stop renderer"""
        self.running = False

