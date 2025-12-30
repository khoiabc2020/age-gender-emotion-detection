"""
DeepSORT Tracker for multi-face tracking
Tuần 5: Face Detection & Tracking pipeline
Improved implementation with IoU matching and Kalman filter-like prediction
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import time


class Track:
    """Single track object"""
    
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], frame_id: int):
        self.track_id = track_id
        self.bbox = bbox  # (x, y, w, h)
        self.frame_id = frame_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history = [bbox]  # Track history for smoothing
        
    def update(self, bbox: Tuple[int, int, int, int], frame_id: int):
        """Update track with new detection"""
        # Smooth bbox using history (simple moving average)
        if len(self.history) > 0:
            prev_bbox = self.history[-1]
            # Weighted average: 70% new, 30% old
            x = int(0.7 * bbox[0] + 0.3 * prev_bbox[0])
            y = int(0.7 * bbox[1] + 0.3 * prev_bbox[1])
            w = int(0.7 * bbox[2] + 0.3 * prev_bbox[2])
            h = int(0.7 * bbox[3] + 0.3 * prev_bbox[3])
            self.bbox = (x, y, w, h)
        else:
            self.bbox = bbox
        
        self.history.append(self.bbox)
        if len(self.history) > 10:  # Keep last 10 frames
            self.history.pop(0)
        
        self.hits += 1
        self.age = 0
        self.time_since_update = 0
        self.frame_id = frame_id
    
    def predict(self):
        """Predict next bbox position (simple linear prediction)"""
        if len(self.history) >= 2:
            # Simple velocity estimation
            prev_bbox = self.history[-2]
            curr_bbox = self.history[-1]
            
            vx = curr_bbox[0] - prev_bbox[0]
            vy = curr_bbox[1] - prev_bbox[1]
            
            # Predict next position
            x = curr_bbox[0] + vx
            y = curr_bbox[1] + vy
            w = curr_bbox[2]
            h = curr_bbox[3]
            
            return (x, y, w, h)
        return self.bbox


class DeepSORTTracker:
    """
    DeepSORT tracker for tracking multiple faces across frames
    Simplified implementation optimized for edge devices
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize DeepSORT tracker
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before track is confirmed
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of (x, y, w, h, confidence) bounding boxes
            
        Returns:
            Dictionary mapping track_id to (x, y, w, h) bounding box
        """
        self.frame_count += 1
        
        # Predict current positions for all tracks
        for track in self.tracks.values():
            track.age += 1
            track.time_since_update += 1
            track.bbox = track.predict()
        
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, self.tracks
        )
        
        # Update matched tracks
        for det_idx, trk_idx in matched:
            track = self.tracks[trk_idx]
            x, y, w, h, conf = detections[det_idx]
            track.update((x, y, w, h), self.frame_count)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            x, y, w, h, conf = detections[det_idx]
            track = Track(self.next_id, (x, y, w, h), self.frame_count)
            self.tracks[self.next_id] = track
            self.next_id += 1
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_age:
                tracks_to_remove.append(track_id)
            elif track.hits < self.min_hits:
                # Remove tracks that haven't been confirmed
                if track.time_since_update > 5:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Return only confirmed tracks
        active_tracks = {}
        for track_id, track in self.tracks.items():
            if track.hits >= self.min_hits:
                active_tracks[track_id] = track.bbox
        
        return active_tracks
    
    def _associate_detections_to_trackers(
        self,
        detections: List[Tuple[int, int, int, int, float]],
        tracks: Dict[int, Track]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to existing tracks using IoU matching
        
        Returns:
            (matched, unmatched_dets, unmatched_trks)
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(tracks.keys())
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        track_ids = list(tracks.keys())
        
        for d, det in enumerate(detections):
            for t, track_id in enumerate(track_ids):
                track = tracks[track_id]
                iou_matrix[d, t] = self._calculate_iou(
                    (det[0], det[1], det[2], det[3]),
                    track.bbox
                )
        
        # Hungarian algorithm-like greedy matching (optimized)
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracks)))
        
        # Greedy matching: sort by IoU descending and match best pairs
        if iou_matrix.size > 0:
            # Create list of (iou, det_idx, trk_idx) and sort descending
            iou_pairs = []
            for d in range(len(detections)):
                for t in range(len(tracks)):
                    if iou_matrix[d, t] >= self.iou_threshold:
                        iou_pairs.append((iou_matrix[d, t], d, t))
            
            # Sort by IoU descending
            iou_pairs.sort(reverse=True, key=lambda x: x[0])
            
            # Match greedily
            used_dets = set()
            used_trks = set()
            for iou_val, d_idx, t_idx in iou_pairs:
                if d_idx not in used_dets and t_idx not in used_trks:
                    matched.append((d_idx, track_ids[t_idx]))
                    used_dets.add(d_idx)
                    used_trks.add(t_idx)
            
            # Update unmatched lists
            unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
            unmatched_trks = [i for i in range(len(tracks)) if i not in used_trks]
        
        return matched, unmatched_dets, unmatched_trks
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate IoU between two bounding boxes
        
        Args:
            box1: (x, y, w, h)
            box2: (x, y, w, h)
            
        Returns:
            IoU value
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
