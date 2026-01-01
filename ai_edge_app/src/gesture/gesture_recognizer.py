"""
Gesture Recognizer - MediaPipe Hands
Giai đoạn 4 Touchless Control
Điều khiển không chạm: Lướt tay trái/phải để điều hướng
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
from enum import Enum

class GestureType(Enum):
    """Gesture types"""
    NONE = "none"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    POINT = "point"
    OPEN_HAND = "open_hand"
    CLOSED_FIST = "closed_fist"

class GestureRecognizer:
    """
    Gesture Recognizer using MediaPipe Hands
    Detects hand gestures for touchless control
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2
    ):
        """
        Initialize Gesture Recognizer
        
        Args:
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            max_num_hands: Maximum number of hands to detect
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Gesture tracking
        self.hand_history: List[Dict] = []
        self.swipe_threshold = 50  # Pixels
        self.swipe_time_threshold = 0.3  # Seconds
        self.last_gesture_time = {}
    
    def detect_hands(self, image: np.ndarray) -> List[Dict]:
        """
        Detect hands in image
        
        Args:
            image: BGR image
        
        Returns:
            List of hand landmarks
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.hands.process(image_rgb)
        
        hands_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_hand_landmarks
            ):
                # Get hand landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                # Get hand type (left/right)
                hand_type = "left" if handedness.classification[0].label == "Left" else "right"
                
                hands_data.append({
                    'landmarks': landmarks,
                    'type': hand_type,
                    'bbox': self._get_hand_bbox(landmarks, image.shape)
                })
        
        return hands_data
    
    def _get_hand_bbox(self, landmarks: List[Dict], image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Get bounding box of hand"""
        xs = [lm['x'] for lm in landmarks]
        ys = [lm['y'] for lm in landmarks]
        
        x_min = int(min(xs) * image_shape[1])
        y_min = int(min(ys) * image_shape[0])
        x_max = int(max(xs) * image_shape[1])
        y_max = int(max(ys) * image_shape[0])
        
        return (x_min, y_min, x_max, y_max)
    
    def recognize_gesture(self, hands_data: List[Dict], frame_time: float) -> GestureType:
        """
        Recognize gesture from hand data
        
        Args:
            hands_data: List of hand data from detect_hands
            frame_time: Current frame time
        
        Returns:
            Detected gesture type
        """
        if not hands_data:
            return GestureType.NONE
        
        # Use first hand
        hand = hands_data[0]
        landmarks = hand['landmarks']
        
        # Get key points
        wrist = landmarks[0]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        thumb_tip = landmarks[4]
        
        # Check if hand is open or closed
        fingers_up = self._count_fingers_up(landmarks)
        
        if fingers_up == 0:
            return GestureType.CLOSED_FIST
        elif fingers_up >= 4:
            return GestureType.OPEN_HAND
        
        # Detect swipe gesture
        if len(self.hand_history) > 0:
            last_hand = self.hand_history[-1]
            if 'landmarks' in last_hand:
                last_wrist = last_hand['landmarks'][0]
                
                # Calculate movement
                dx = (wrist['x'] - last_wrist['x']) * 640  # Assume 640 width
                dy = (wrist['y'] - last_wrist['y']) * 480  # Assume 480 height
                
                # Check swipe
                if abs(dx) > self.swipe_threshold and abs(dx) > abs(dy):
                    if dx > 0:
                        return GestureType.SWIPE_RIGHT
                    else:
                        return GestureType.SWIPE_LEFT
                elif abs(dy) > self.swipe_threshold and abs(dy) > abs(dx):
                    if dy > 0:
                        return GestureType.SWIPE_DOWN
                    else:
                        return GestureType.SWIPE_UP
        
        # Store current hand for next frame
        self.hand_history.append({
            'landmarks': landmarks,
            'time': frame_time
        })
        
        # Keep only recent history
        if len(self.hand_history) > 10:
            self.hand_history.pop(0)
        
        return GestureType.POINT
    
    def _count_fingers_up(self, landmarks: List[Dict]) -> int:
        """Count number of fingers up"""
        # Thumb
        thumb_up = landmarks[4]['x'] > landmarks[3]['x']
        
        # Other fingers
        fingers = [
            landmarks[8]['y'] < landmarks[6]['y'],  # Index
            landmarks[12]['y'] < landmarks[10]['y'],  # Middle
            landmarks[16]['y'] < landmarks[14]['y'],  # Ring
            landmarks[20]['y'] < landmarks[18]['y']  # Pinky
        ]
        
        return sum(fingers) + (1 if thumb_up else 0)
    
    def draw_hands(self, image: np.ndarray, hands_data: List[Dict]) -> np.ndarray:
        """Draw hand landmarks on image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe format
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        # Create hand landmarks object
        hand_landmarks_list = []
        for hand in hands_data:
            landmarks = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
            for lm in hand['landmarks']:
                landmark = landmarks.landmark.add()
                landmark.x = lm['x']
                landmark.y = lm['y']
                landmark.z = lm['z']
            hand_landmarks_list.append(landmarks)
        
        # Draw
        annotated_image = image_rgb.copy()
        for hand_landmarks in hand_landmarks_list:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    # Test
    recognizer = GestureRecognizer()
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    hands = recognizer.detect_hands(test_image)
    gesture = recognizer.recognize_gesture(hands, 0.0)
    print(f"Detected gesture: {gesture.value}")

