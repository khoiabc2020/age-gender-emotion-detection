"""
ByteTrack Tracker - SOTA Multi-Object Tracking
Giai đoạn 3 Tuần 7: Business Logic & Tracking
Thay thế DeepSORT với ByteTrack (nhẹ hơn, chính xác hơn)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import time


class Track:
    """Track object for ByteTrack"""
    
    def __init__(self, bbox: np.ndarray, track_id: int, score: float, frame_id: int):
        """
        Initialize track
        
        Args:
            bbox: [x1, y1, x2, y2]
            track_id: Unique track ID
            score: Detection confidence
            frame_id: Current frame ID
        """
        self.bbox = bbox.astype(np.float32)
        self.track_id = track_id
        self.score = score
        self.frame_id = frame_id
        self.state = 'tentative'  # tentative, confirmed, deleted
        self.time_since_update = 0
        self.history: List[np.ndarray] = [bbox.copy()]
        
        # Kalman filter states (simplified)
        self.velocity = np.zeros(4, dtype=np.float32)
    
    def update(self, bbox: np.ndarray, score: float, frame_id: int):
        """Update track with new detection"""
        # Update velocity (simple linear motion model)
        self.velocity = 0.7 * self.velocity + 0.3 * (bbox - self.bbox)
        self.bbox = bbox.astype(np.float32)
        self.score = score
        self.frame_id = frame_id
        self.time_since_update = 0
        self.history.append(bbox.copy())
        
        if self.state == 'tentative':
            self.state = 'confirmed'
    
    def predict(self) -> np.ndarray:
        """Predict next bbox position"""
        return self.bbox + self.velocity
    
    def mark_missed(self):
        """Mark track as missed"""
        self.time_since_update += 1


class ByteTracker:
    """
    ByteTrack Multi-Object Tracker
    Lightweight and accurate tracking algorithm
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        high_thresh: float = 0.6,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
        track_buffer: int = 30,
        min_box_area: float = 10.0
    ):
        """
        Initialize ByteTracker
        
        Args:
            track_thresh: Detection confidence threshold
            high_thresh: High confidence threshold
            match_thresh: IoU threshold for matching
            frame_rate: Video frame rate
            track_buffer: Buffer frames for track recovery
            min_box_area: Minimum box area
        """
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        
        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []
        self.frame_id = 0
        self.next_id = 1
    
    def iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections with keys: 'bbox', 'score', 'class'
        
        Returns:
            List of tracks with 'track_id', 'bbox', 'score', 'state'
        """
        self.frame_id += 1
        
        # Filter detections by threshold and area
        valid_detections = []
        for det in detections:
            bbox = det['bbox']
            score = det['score']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            if score >= self.track_thresh and area >= self.min_box_area:
                valid_detections.append(det)
        
        # Separate high and low confidence detections
        high_conf_dets = [d for d in valid_detections if d['score'] >= self.high_thresh]
        low_conf_dets = [d for d in valid_detections if d['score'] < self.high_thresh]
        
        # Update confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.state == 'confirmed']
        tentative_tracks = [t for t in self.tracks if t.state == 'tentative']
        
        # Match high confidence detections with tracks
        matched_tracks, unmatched_tracks, unmatched_dets = self._match_detections_to_tracks(
            high_conf_dets, confirmed_tracks + tentative_tracks
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            track = (confirmed_tracks + tentative_tracks)[track_idx]
            det = high_conf_dets[det_idx]
            track.update(det['bbox'], det['score'], self.frame_id)
        
        # Create new tracks from unmatched high confidence detections
        for det_idx in unmatched_dets:
            det = high_conf_dets[det_idx]
            new_track = Track(det['bbox'], self.next_id, det['score'], self.frame_id)
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Match low confidence detections with unmatched tracks
        unmatched_confirmed = [confirmed_tracks[i] for i in unmatched_tracks if i < len(confirmed_tracks)]
        if unmatched_confirmed and low_conf_dets:
            matched_tracks_low, _, _ = self._match_detections_to_tracks(
                low_conf_dets, unmatched_confirmed
            )
            for track_idx, det_idx in matched_tracks_low:
                track = unmatched_confirmed[track_idx]
                det = low_conf_dets[det_idx]
                track.update(det['bbox'], det['score'], self.frame_id)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            track = (confirmed_tracks + tentative_tracks)[track_idx]
            track.mark_missed()
        
        # Update lost tracks
        for track in self.tracks:
            if track.time_since_update > 0:
                self.lost_tracks.append(track)
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update == 0]
        
        # Recover lost tracks
        if self.lost_tracks:
            lost_bboxes = [t.predict() for t in self.lost_tracks]
            matched_lost, _, _ = self._match_detections_to_tracks(
                [{'bbox': bbox} for bbox in lost_bboxes],
                self.lost_tracks
            )
            for track_idx, _ in matched_lost:
                track = self.lost_tracks[track_idx]
                track.time_since_update = 0
                self.tracks.append(track)
        
        # Remove old lost tracks
        self.lost_tracks = [
            t for t in self.lost_tracks
            if t.time_since_update <= self.track_buffer
        ]
        
        # Return active tracks
        active_tracks = []
        for track in self.tracks:
            if track.state == 'confirmed':
                active_tracks.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'score': track.score,
                    'state': track.state
                })
        
        return active_tracks
    
    def _match_detections_to_tracks(
        self,
        detections: List[Dict],
        tracks: List[Track]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using IoU
        
        Returns:
            (matched_pairs, unmatched_tracks, unmatched_detections)
        """
        if not detections or not tracks:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self.iou(track.bbox, det['bbox'])
        
        # Greedy matching
        matched_pairs = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        # Sort by IoU descending
        matches = []
        for i in range(len(tracks)):
            for j in range(len(detections)):
                if iou_matrix[i, j] >= self.match_thresh:
                    matches.append((i, j, iou_matrix[i, j]))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        
        used_tracks = set()
        used_dets = set()
        
        for i, j, _ in matches:
            if i not in used_tracks and j not in used_dets:
                matched_pairs.append((i, j))
                used_tracks.add(i)
                used_dets.add(j)
        
        unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in used_dets]
        
        return matched_pairs, unmatched_tracks, unmatched_dets


if __name__ == "__main__":
    # Test
    tracker = ByteTracker()
    
    # Simulate detections
    detections = [
        {'bbox': np.array([100, 100, 200, 200]), 'score': 0.8, 'class': 0},
        {'bbox': np.array([300, 300, 400, 400]), 'score': 0.7, 'class': 0}
    ]
    
    tracks = tracker.update(detections)
    print(f"Tracks: {len(tracks)}")
    for track in tracks:
        print(f"  ID: {track['track_id']}, Bbox: {track['bbox']}")

