"""
Edge AI Application for Smart Retail Analytics.

Real-time face detection, tracking, attribute recognition, and personalized
advertisement recommendation system optimized for edge computing.
"""

import cv2
import json
import time
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from collections import deque

from src.detectors import RetinaFaceDetector, YOLOFaceDetector, YOLOPersonDetector
from src.trackers import DeepSORTTracker, ByteTracker
from src.classifiers import MultiTaskClassifier
from src.ads_engine import AdsSelector
from src.services.generative_ads import GenerativeAds
from src.services.qr_generator import QRCodeService
from src.core.anti_spoofing import MiniFASNet
from src.core.face_restoration import FaceRestorer
from src.core.dwell_time import DwellTimeTracker
from src.utils import MQTTClient, setup_logger

class EdgeAIApp:
    """Main application class for edge AI processing.
    
    Optimized for real-time performance with multi-threading support.
    Handles face detection, tracking, attribute classification, and
    personalized advertisement selection.
    
    Attributes:
        config: Application configuration dictionary
        logger: Application logger instance
        detector: Face detection model
        tracker: Object tracking system
        classifier: Attribute classification model
        ads_selector: Advertisement recommendation engine
        mqtt_client: MQTT client for cloud communication
    """
    
    def __init__(self, config_path: str = "configs/camera_config.json"):
        """Initialize Edge AI Application.
        
        Args:
            config_path: Path to configuration JSON file
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Setup logger
        self.logger = setup_logger(
            'EdgeAIApp', 
            log_file='logs/edge_app.log'
        )
        
        # Initialize components
        self.detector = None
        self.tracker = None
        self.classifier = None
        self.ads_selector = None
        self.mqtt_client = None
        
        # Advanced Modules ()
        # Advanced modules
        self.anti_spoofing = None
        self.face_restorer = None
        
        # Business logic
        self.dwell_tracker = None
        
        # Services
        self.gen_ads = None
        self.qr_service = None
        
        # Camera
        self.camera = None
        self.running = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps_history = deque(maxlen=30)
        
        # Threading for performance ()
        self.frame_queue = deque(maxlen=2)  # Buffer for frames
        self.processing_lock = threading.Lock()
        self.display_frame = None
        
        # Track processed faces to avoid redundant processing
        # Thread-safe dictionaries with locks
        self.processed_tracks: Dict[int, Dict] = {}  # Track processing history
        self.track_attributes: Dict[int, Dict] = {}  # Store attributes per track
        self.tracks_lock = threading.Lock()  # Lock for track dictionaries
        
        # Cleanup old tracks periodically
        self.last_cleanup_time = time.time()
        self.track_cleanup_interval = 2.0  # Check more frequently for feedback

    
    def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("Initializing Edge AI Application...")
            
            # Initialize detector (support multiple backends: RetinaFace, YOLO)
            model_dir = Path("models")
            detector_type = self.config['detection'].get('type', 'retinaface')  # 'retinaface', 'yolo_face', 'yolo_person'
            
            if detector_type == 'yolo_face':
                # YOLO Face Detection
                yolo_model = model_dir / "yolov8n-face.onnx"  # or yolov5-face.onnx
                if yolo_model.exists():
                    self.detector = YOLOFaceDetector(
                        str(yolo_model),
                        confidence_threshold=self.config['detection']['confidence_threshold'],
                        iou_threshold=self.config['detection'].get('iou_threshold', 0.45)
                    )
                    self.logger.info("Loaded YOLO Face Detection model")
                else:
                    self.logger.warning(f"YOLO face model not found: {yolo_model}, falling back to RetinaFace")
                    detector_type = 'retinaface'  # Fallback
            
            elif detector_type == 'yolo_person':
                # YOLO Person Detection (full body)
                yolo_model = model_dir / "yolov8n.onnx"  # COCO pretrained
                if yolo_model.exists():
                    self.detector = YOLOPersonDetector(
                        str(yolo_model),
                        confidence_threshold=self.config['detection']['confidence_threshold'],
                        iou_threshold=self.config['detection'].get('iou_threshold', 0.45)
                    )
                    self.logger.info("Loaded YOLO Person Detection model (full body)")
                else:
                    self.logger.warning(f"YOLO person model not found: {yolo_model}, falling back to RetinaFace")
                    detector_type = 'retinaface'  # Fallback
            
            # Default: RetinaFace
            if detector_type == 'retinaface' or self.detector is None:
                detector_model = model_dir / "retinaface_mnet.onnx"
                
                if detector_model.exists():
                    self.detector = RetinaFaceDetector(
                        str(detector_model),
                        confidence_threshold=self.config['detection']['confidence_threshold']
                    )
                    self.logger.info("Loaded RetinaFace ONNX model")
                else:
                    self.logger.warning(f"Detector model not found: {detector_model}, using fallback")
                    self.detector = RetinaFaceDetector(
                        confidence_threshold=self.config['detection']['confidence_threshold']
                    )
            
            # Initialize tracker (ByteTrack thay DeepSORT)
            use_bytetrack = self.config['tracking'].get('use_bytetrack', False)
            if use_bytetrack:
                self.tracker = ByteTracker(
                    track_thresh=self.config['tracking'].get('track_thresh', 0.5),
                    high_thresh=self.config['tracking'].get('high_thresh', 0.6),
                    match_thresh=self.config['tracking'].get('iou_threshold', 0.3),
                    frame_rate=self.config['camera'].get('fps', 30)
                )
                self.logger.info("Initialized ByteTrack tracker")
            else:
                self.tracker = DeepSORTTracker(
                    max_age=self.config['tracking']['max_age'],
                    min_hits=self.config['tracking']['min_hits'],
                    iou_threshold=self.config['tracking']['iou_threshold']
                )
                self.logger.info("Initialized DeepSORT tracker")
            
            # Initialize Dwell Time Tracker ()
            dwell_threshold = self.config['tracking'].get('dwell_threshold', 3.0)
            self.dwell_tracker = DwellTimeTracker(threshold=dwell_threshold)
            self.logger.info(f"Initialized Dwell Time Tracker (threshold: {dwell_threshold}s)")
            
            # Initialize classifier
            classifier_model = model_dir / "multitask_efficientnet.onnx"
            if classifier_model.exists():
                self.classifier = MultiTaskClassifier(str(classifier_model))
                self.logger.info("Loaded Multi-task classifier ONNX model")
            else:
                self.logger.warning(f"Classifier model not found: {classifier_model}")
                self.logger.warning("Please copy model from training_experiments/models/")
            

            # Initialize ads selector
            ads_rules_path = Path("configs/ads_rules.json")
            if ads_rules_path.exists():
                self.ads_selector = AdsSelector(str(ads_rules_path))
                self.logger.info("Loaded ads rules & LinUCB")
            else:
                self.logger.warning(f"Ads rules not found: {ads_rules_path}")
                
            # Initialize Advanced Modules ()
            model_dir = Path("models")
            anti_spoofing_model = model_dir / "minifasnet.onnx"
            self.anti_spoofing = MiniFASNet(
                model_path=str(anti_spoofing_model) if anti_spoofing_model.exists() else None
            )
            self.logger.info("Initialized Anti-Spoofing (MiniFASNet)")
            
            face_restore_model = model_dir / "espcn.onnx"
            self.face_restorer = FaceRestorer(
                model_path=str(face_restore_model) if face_restore_model.exists() else None,
                method="espcn"  # Lightweight option
            )
            self.logger.info("Initialized Face Restoration (ESPCN)")
            
            # Initialize GenAI and QR Services
            self.gen_ads = GenerativeAds() # Will read env var GEMINI_API_KEY
            self.qr_service = QRCodeService()
            self.logger.info("Initialized GenAI & QR Services")
            
            # Initialize MQTT client (optional, won't fail if not available)
            mqtt_config = self.config['mqtt']
            try:
                self.mqtt_client = MQTTClient(
                    broker=mqtt_config['broker'],
                    port=mqtt_config['port'],
                    topic=mqtt_config['topic'],
                    device_key=mqtt_config['device_key']
                )
                if self.mqtt_client.connect():
                    self.logger.info("Connected to MQTT broker")
                else:
                    self.logger.warning("MQTT connection failed, continuing without MQTT")
                    self.mqtt_client = None
            except Exception as e:
                self.logger.warning(f"MQTT initialization failed: {e}, continuing without MQTT")
                self.mqtt_client = None
            
            # Initialize camera
            cam_config = self.config['camera']
            self.camera = cv2.VideoCapture(cam_config['source'])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config['height'])
            
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera source: {cam_config['source']}")
            
            self.logger.info("Camera initialized successfully")
            self.logger.info("Initialization complete!")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            raise
    
    def run(self):
        """Main processing loop with threading optimization"""
        if not self.camera:
            self.logger.error("Camera not initialized")
            return
        
        self.running = True
        self.logger.info("Starting main processing loop...")
        
        # Performance optimization: Frame skipping
        frame_skip = 4  # Process every 4th frame (reduce CPU load by 75%)
        frame_count = 0
        target_fps = 10  # Target 10 FPS instead of 30 (reduce CPU by 66%)
        frame_delay = 1.0 / target_fps  # ~100ms per frame
        
        # Classification optimization: Only classify every N seconds per track
        classification_interval = 3.0  # Classify each track every 3 seconds
        last_classification_time = {}  # Track last classification time per track_id
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)  # Small delay before retry
                    continue
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % (frame_skip + 1) != 0:
                    time.sleep(frame_delay)
                    continue
                
                start_time = time.time()
                
                
                detections = []
                if self.detector:
                    try:
                        raw_detections = self.detector.detect(frame)
                        # Convert detector output to tracker format
                        # Detector returns List[Tuple[x, y, w, h, score]] or List[Tuple[x1, y1, x2, y2, score]]
                        # Tracker expects List[Dict] with keys: 'bbox', 'score', 'class'
                        for det in raw_detections:
                            if isinstance(det, tuple):
                                # Tuple format: (x, y, w, h, score) or (x1, y1, x2, y2, score)
                                if len(det) >= 5:
                                    x, y, w_or_x2, h_or_y2, score = det[0], det[1], det[2], det[3], det[4]
                                    # Determine if it's (x, y, w, h) or (x1, y1, x2, y2)
                                    if w_or_x2 > 100 or h_or_y2 > 100:
                                        # Likely (x, y, w, h) format
                                        bbox = np.array([x, y, x + w_or_x2, y + h_or_y2], dtype=np.float32)
                                    else:
                                        # Likely (x1, y1, x2, y2) format
                                        bbox = np.array([x, y, w_or_x2, h_or_y2], dtype=np.float32)
                                    detections.append({
                                        'bbox': bbox,
                                        'score': float(score),
                                        'class': 0  # Face class
                                    })
                            elif isinstance(det, dict):
                                # Already in correct format
                                detections.append(det)
                    except Exception as e:
                        self.logger.warning(f"Detection error: {e}")
                        detections = []
                
                # Step 2: Tracking (ByteTrack hoặc DeepSORT)
                tracker_result = self.tracker.update(detections)
                
                # Normalize tracker output format
                # ByteTracker returns List[Dict], DeepSORT returns Dict[int, Tuple]
                if isinstance(tracker_result, list):
                    # ByteTracker format: [{'track_id', 'bbox', 'score', 'state'}, ...]
                    active_tracks = {}
                    for track in tracker_result:
                        track_id = track['track_id']
                        bbox_array = track['bbox']  # [x1, y1, x2, y2] or [x, y, w, h]
                        # Convert to (x, y, w, h) format
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
                                active_tracks[track_id] = (x, y, w, h)
                else:
                    # DeepSORT format: Dict[int, Tuple[int, int, int, int]]
                    active_tracks = tracker_result
                
                # Step 3: Classification and Ad Selection
                # Use frame directly instead of copy to save memory
                display_frame = frame
                
                # Update Dwell Time ()
                current_time = time.time()
                if self.dwell_tracker:
                    for track_id in active_tracks.keys():
                        self.dwell_tracker.update_track(track_id, current_time)
                
                # Cleanup old tracks periodically
                if current_time - self.last_cleanup_time > self.track_cleanup_interval:
                    self._cleanup_old_tracks(current_time)
                    self.last_cleanup_time = current_time
                
                for track_id, bbox in active_tracks.items():
                    x, y, w, h = bbox
                    
                    # Ensure valid coordinates
                    x = max(0, min(x, frame.shape[1] - 1))
                    y = max(0, min(y, frame.shape[0] - 1))
                    w = max(1, min(w, frame.shape[1] - x))
                    h = max(1, min(h, frame.shape[0] - y))
                    
                    # Skip if recently processed (to reduce computation)
                    # Process every N seconds per track to balance accuracy and performance
                    # Optimization: Use cached attributes to reduce computation
                    should_skip = False
                    with self.tracks_lock:
                        if track_id in self.processed_tracks:
                            last_time = self.processed_tracks[track_id].get('last_processed', 0)
                            # Only classify every 3 seconds per track (reduced from 2s)
                            if current_time - last_time < classification_interval:
                                # Use cached attributes
                                if track_id in self.track_attributes:
                                    attributes = self.track_attributes[track_id].copy()
                                    ad = self.processed_tracks[track_id].get('ad')
                                    self._draw_track_info(display_frame, track_id, bbox, attributes, ad)
                                    should_skip = True
                    
                    if should_skip:
                        continue
                    
                    # Update last classification time
                    last_classification_time[track_id] = current_time
                    
                    # Crop face region with padding to avoid edge issues
                    # Add small padding to ensure we capture full face
                    padding = 5
                    x_pad = max(0, x - padding)
                    y_pad = max(0, y - padding)
                    w_pad = min(frame.shape[1] - x_pad, w + 2 * padding)
                    h_pad = min(frame.shape[0] - y_pad, h + 2 * padding)
                    
                    face_roi = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                    
                    # Dwell Time check - Chỉ xử lý nếu là valid customer (> 3s)
                    if self.dwell_tracker:
                        if not self.dwell_tracker.is_valid_customer(track_id):
                            # Chưa đủ dwell time, chỉ vẽ basic track
                            dwell_time = self.dwell_tracker.get_dwell_time(track_id)
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (128, 128, 128), 2)
                            cv2.putText(display_frame, f"ID:{track_id} ({dwell_time:.1f}s)", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                            continue
                    
                    # Validate face ROI size (minimum 32x32 for model input)
                    if (face_roi.size > 0 and 
                        len(face_roi.shape) == 3 and
                        face_roi.shape[0] >= 32 and 
                        face_roi.shape[1] >= 32 and 
                        self.classifier):
                        try:
                            # Advanced Modules
                            # Step 1: Anti-Spoofing - Check if face is real
                            if self.anti_spoofing:
                                is_real, spoof_confidence = self.anti_spoofing.predict(face_roi)
                                if not is_real:
                                    # Skip fake faces (photo, video, mask)
                                    self.logger.debug(f"Track {track_id}: Fake face detected (confidence: {spoof_confidence:.2f})")
                                    continue
                            
                            # Step 2: Face Restoration - Enhance low-quality faces
                            if self.face_restorer:
                                face_roi = self.face_restorer.enhance_if_needed(face_roi, quality_threshold=0.3)
                            
                            
                            try:
                                attributes = self.classifier.predict(face_roi)
                                
                                # Update processed tracks with new attributes
                                with self.tracks_lock:
                                    if track_id not in self.processed_tracks:
                                        self.processed_tracks[track_id] = {}
                                    self.processed_tracks[track_id]['last_processed'] = current_time
                                    self.track_attributes[track_id] = attributes
                            except Exception as e:
                                self.logger.warning(f"Classification error for track {track_id}: {e}")
                                continue
                            
                            # Step 3b: Select advertisement (Ad Recommendation Engine)
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
                                    self.logger.warning(f"Ad selection error for track {track_id}: {e}")
                                    ad = None
                                
                                # Step 3c: Add Generative Content (Slogan & QR)
                                if ad:
                                    # Generate Slogan (Async-like or fast check)
                                    if self.gen_ads:
                                        ad['slogan'] = self.gen_ads.generate_slogan(
                                            ad['name'], attributes['emotion'], 
                                            attributes['age'], attributes['gender']
                                        )
                                    
                                    # Generate QR (Voucher)
                                    if self.qr_service:
                                        qr_img, code = self.qr_service.get_voucher_qr(ad['ad_id'])
                                        ad['qr_code'] = qr_img
                                        ad['voucher_code'] = code
                            
                            # Store attributes (thread-safe)
                            with self.tracks_lock:
                                self.track_attributes[track_id] = attributes.copy()
                                # Maintain start_time if track exists, else set current
                                old_data = self.processed_tracks.get(track_id, {})
                                start_t = old_data.get('start_time', time.time())
                                
                                self.processed_tracks[track_id] = {
                                    'last_processed': time.time(),
                                    'start_time': start_t,
                                    'ad': ad
                                }
                            
                            # Log analytics
                            self.logger.info(
                                f"Track {track_id}: Age={attributes['age']}, "
                                f"Gender={attributes['gender']}, "
                                f"Emotion={attributes['emotion']}, "
                                f"Ad={ad['ad_id'] if ad else 'N/A'}"
                            )
                            
                            # Send analytics to cloud (async)
                            if self.mqtt_client and ad:
                                try:
                                    self.mqtt_client.publish_analytics(
                                        track_id=track_id,
                                        age=attributes['age'],
                                        gender=attributes['gender'],
                                        emotion=attributes['emotion'],
                                        ad_id=ad['ad_id']
                                    )
                                except Exception as e:
                                    self.logger.debug(f"MQTT publish failed: {e}")
                            
                            # Draw on display frame
                            self._draw_track_info(display_frame, track_id, bbox, attributes, ad)
                            
                        except Exception as e:
                            self.logger.error(f"Error processing track {track_id}: {e}")
                    else:
                        # Draw basic track info even without classification
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(
                            display_frame, 
                            f"ID: {track_id}", 
                            (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0), 
                            2
                        )
                
                # Calculate and display FPS
                processing_time = time.time() - start_time
                fps = 1.0 / processing_time if processing_time > 0 else 0
                self.fps_history.append(fps)
                
                # Calculate average FPS (more efficient)
                avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
                
                # Display FPS and stats
                fps_color = (0, 255, 0) if avg_fps >= 15 else (0, 165, 255)  # Green if >= 15, Orange if < 15
                cv2.putText(
                    display_frame,
                    f"FPS: {avg_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    fps_color,
                    2
                )
                
                # Display active tracks count
                cv2.putText(
                    display_frame,
                    f"Tracks: {len(active_tracks)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Display frame
                cv2.imshow('Edge AI App - Smart Retail Analytics', display_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                
                # Performance: Add delay to maintain target FPS and prevent freezing
                elapsed = time.time() - start_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    
    def _draw_track_info(
        self, 
        frame: np.ndarray, 
        track_id: int, 
        bbox: tuple, 
        attributes: Dict,
        ad: Optional[Dict] = None
    ):
        """Draw tracking, attribute information, and interactive ads on frame"""
        x, y, w, h = bbox
        
        # Draw bounding box
        color = (0, 255, 0)
        if ad: color = (255, 100, 0) # Blue-ish for ad active
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Prepare text lines
        info_lines = [
            f"ID: {track_id}",
            f"Age: {attributes['age']} - {attributes['gender']}",
            f"Emotion: {attributes['emotion']}"
        ]
        
        # Interactive Ad Overlay
        if ad:
            # 1. Main Ad Title
            info_lines.append(f"AD: {ad.get('name', 'Promotion')}")
            
            # 2. Generative Slogan (if available)
            slogan = ad.get('slogan')
            if slogan:
                # Wrap text if too long
                info_lines.append(f"\"{slogan}\"")
                
            # 3. QR Code Overlay (Top Right of BBox or Side)
            qr_img = ad.get('qr_code')
            if isinstance(qr_img, np.ndarray):
                # Calculate position (Top-Right of face)
                qr_size = qr_img.shape[0]
                qr_x = min(frame.shape[1] - qr_size, x + w + 10)
                qr_y = max(0, y)
                
                # Check boundaries
                if qr_x + qr_size <= frame.shape[1] and qr_y + qr_size <= frame.shape[0]:
                    # Overlay QR
                    frame[qr_y:qr_y+qr_size, qr_x:qr_x+qr_size] = qr_img
                    # "Scan Me" text
                    cv2.putText(frame, "SCAN ME", (qr_x, qr_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw text background and text
        text_y = y - 10
        for i, line in enumerate(reversed(info_lines)):
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            curr_y = text_y - i * 25
            
            # Background
            cv2.rectangle(
                frame,
                (x, curr_y - 20),
                (x + text_size[0] + 10, curr_y + 5),
                (0, 0, 0),
                -1
            )
            # Text
            cv2.putText(
                frame,
                line,
                (x + 5, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
    
    def _cleanup_old_tracks(self, current_time: float):
        """Cleanup old tracks and send feedback to Recommender System ()"""
        with self.tracks_lock:
            # Remove tracks older than 5 seconds (lost attention)
            max_age = 5.0 
            tracks_to_remove = []
            
            for track_id, track_data in self.processed_tracks.items():
                last_seen = track_data.get('last_processed', 0)
                if current_time - last_seen > max_age:
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                # Calculate Duration for Feedback
                track_data = self.processed_tracks[track_id]
                start_time = track_data.get('start_time', track_data.get('last_processed')) # Fallback
                duration = current_time - start_time
                
                # Get final dwell time
                final_dwell_time = None
                if self.dwell_tracker:
                    final_dwell_time = self.dwell_tracker.remove_track(track_id)
                
                # Send Feedback to LinUCB (Ad Recommendation Engine)
                if self.ads_selector and duration > 0:
                    self.ads_selector.update_feedback(track_id, duration)
                    self.logger.info(f"Feedback Sent - Track {track_id}: Duration {duration:.2f}s, Dwell {final_dwell_time:.2f}s" if final_dwell_time else f"Feedback Sent - Track {track_id}: Duration {duration:.2f}s")

                self.processed_tracks.pop(track_id, None)
                self.track_attributes.pop(track_id, None)
            
            # Cleanup dwell tracker old tracks
            if self.dwell_tracker:
                self.dwell_tracker.cleanup_old_tracks(current_time, max_age)
            
            if tracks_to_remove:
                self.logger.debug(f"Cleaned up {len(tracks_to_remove)} old tracks")
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        self.running = False
        
        # Cleanup with lock
        with self.tracks_lock:
            self.processed_tracks.clear()
            self.track_attributes.clear()
        
        if self.camera:
            self.camera.release()
        
        if self.mqtt_client:
            self.mqtt_client.disconnect()
        
        cv2.destroyAllWindows()
        self.logger.info("Cleanup complete")

def main():
    """Main entry point"""
    app = EdgeAIApp()
    try:
        app.initialize()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
