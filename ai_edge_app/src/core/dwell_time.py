"""
Dwell Time Logic
Business Logic & Tracking
Chỉ tính là "khách hàng" nếu nhìn vào cam > 3 giây
"""

import time
from typing import Dict, Optional
from collections import defaultdict

class DwellTimeTracker:
    """
    Track dwell time for each customer
    Only count as "customer" if looking at camera > threshold seconds
    """
    
    def __init__(self, threshold: float = 3.0):
        """
        Initialize Dwell Time Tracker
        
        Args:
            threshold: Minimum dwell time in seconds to count as customer
        """
        self.threshold = threshold
        self.track_start_times: Dict[int, float] = {}  # track_id -> start_time
        self.track_last_seen: Dict[int, float] = {}  # track_id -> last_seen_time
        self.valid_customers: Dict[int, bool] = {}  # track_id -> is_valid_customer
        self.dwell_times: Dict[int, float] = {}  # track_id -> total_dwell_time
    
    def update_track(self, track_id: int, current_time: float):
        """
        Update track with current time
        
        Args:
            track_id: Track ID
            current_time: Current timestamp
        """
        # Initialize if new track
        if track_id not in self.track_start_times:
            self.track_start_times[track_id] = current_time
            self.valid_customers[track_id] = False
        
        # Update last seen
        self.track_last_seen[track_id] = current_time
        
        # Calculate dwell time
        start_time = self.track_start_times[track_id]
        dwell_time = current_time - start_time
        
        # Check if valid customer (dwell time > threshold)
        if dwell_time >= self.threshold:
            self.valid_customers[track_id] = True
        
        self.dwell_times[track_id] = dwell_time
    
    def is_valid_customer(self, track_id: int) -> bool:
        """
        Check if track is a valid customer (dwell time > threshold)
        
        Args:
            track_id: Track ID
            
        Returns:
            True if valid customer
        """
        return self.valid_customers.get(track_id, False)
    
    def get_dwell_time(self, track_id: int) -> float:
        """
        Get current dwell time for track
        
        Args:
            track_id: Track ID
            
        Returns:
            Dwell time in seconds
        """
        return self.dwell_times.get(track_id, 0.0)
    
    def remove_track(self, track_id: int) -> Optional[float]:
        """
        Remove track and return final dwell time
        
        Args:
            track_id: Track ID
            
        Returns:
            Final dwell time or None if track not found
        """
        if track_id in self.dwell_times:
            final_dwell_time = self.dwell_times[track_id]
            self.track_start_times.pop(track_id, None)
            self.track_last_seen.pop(track_id, None)
            self.valid_customers.pop(track_id, None)
            self.dwell_times.pop(track_id, None)
            return final_dwell_time
        return None
    
    def cleanup_old_tracks(self, current_time: float, max_age: float = 5.0):
        """
        Cleanup tracks that haven't been seen for a while
        
        Args:
            current_time: Current timestamp
            max_age: Maximum age in seconds before cleanup
        """
        tracks_to_remove = []
        for track_id, last_seen in self.track_last_seen.items():
            if current_time - last_seen > max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self.remove_track(track_id)
        
        return len(tracks_to_remove)
    
    def get_valid_customer_count(self) -> int:
        """Get count of valid customers (dwell time > threshold)"""
        return sum(1 for is_valid in self.valid_customers.values() if is_valid)
    
    def get_all_customer_count(self) -> int:
        """Get count of all tracked customers"""
        return len(self.track_start_times)

