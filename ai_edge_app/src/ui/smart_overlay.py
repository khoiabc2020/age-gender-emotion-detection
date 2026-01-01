"""
Smart Overlay - Rounded Bounding Boxes với Emotion Colors
Real-time Visualization
Bounding box bo tròn, màu theo cảm xúc
"""

from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath
from PyQt6.QtCore import Qt
from typing import Dict, List, Optional
import math

class SmartOverlay:
    """
    Smart Overlay Renderer
    Vẽ bounding boxes bo tròn với màu theo cảm xúc
    """
    
    # Emotion colors (Windows 11 style)
    EMOTION_COLORS = {
        'happy': QColor(16, 124, 16),      # Green
        'sad': QColor(0, 120, 215),       # Blue
        'angry': QColor(232, 17, 35),     # Red
        'surprise': QColor(255, 185, 0),  # Orange/Yellow
        'fear': QColor(118, 118, 118),    # Gray
        'neutral': QColor(200, 200, 200), # Light Gray
        'disgust': QColor(232, 17, 35)    # Red (same as angry)
    }
    
    def __init__(self):
        """Initialize Smart Overlay"""
        self.corner_radius = 12  # Rounded corner radius
        self.line_width = 3
        self.font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        self.small_font = QFont("Segoe UI", 8)
    
    def draw_track_overlay(
        self,
        painter: QPainter,
        x: int,
        y: int,
        w: int,
        h: int,
        track_data: Dict,
        scale_x: float = 1.0,
        scale_y: float = 1.0
    ):
        """
        Draw smart overlay for a track
        
        Args:
            painter: QPainter instance
            x, y, w, h: Scaled coordinates
            track_data: Track data dict with bbox, attributes, etc.
            scale_x, scale_y: Scale factors for coordinate conversion
        """
        # Get emotion color
        emotion = track_data.get('emotion', 'neutral').lower()
        color = self.EMOTION_COLORS.get(emotion, self.EMOTION_COLORS['neutral'])
        
        # Draw rounded rectangle
        self._draw_rounded_rect(painter, x, y, w, h, color)
        
        # Draw label background
        label_bg_height = 30
        label_y = y - label_bg_height
        
        # Rounded label background
        label_path = QPainterPath()
        label_path.addRoundedRect(
            x, label_y, w, label_bg_height,
            self.corner_radius, self.corner_radius
        )
        
        # Semi-transparent background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 200)))
        painter.drawPath(label_path)
        
        # Draw text
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setFont(self.font)
        
        # Track ID
        track_id = track_data.get('track_id', 0)
        painter.drawText(x + 8, label_y + 18, f"ID: {track_id}")
        
        # Age & Gender
        age = track_data.get('age', 0)
        gender = track_data.get('gender', 'Unknown')
        gender_text = "M" if gender.lower() == 'male' else "F" if gender.lower() == 'female' else "?"
        painter.setFont(self.small_font)
        painter.drawText(x + 8, label_y + 30, f"{age}yo {gender_text}")
        
        # Emotion badge (small circle)
        badge_size = 12
        badge_x = x + w - badge_size - 8
        badge_y = y + 8
        
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(badge_x, badge_y, badge_size, badge_size)
        
        # Emotion text
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setFont(self.small_font)
        emotion_text = emotion.capitalize()
        text_width = painter.fontMetrics().boundingRect(emotion_text).width()
        painter.drawText(
            badge_x - text_width - 5,
            badge_y + badge_size - 2,
            emotion_text
        )
    
    def _draw_rounded_rect(
        self,
        painter: QPainter,
        x: int,
        y: int,
        w: int,
        h: int,
        color: QColor
    ):
        """
        Draw rounded rectangle với gradient effect
        
        Args:
            painter: QPainter instance
            x, y, w, h: Rectangle coordinates
            color: Border color
        """
        # Create rounded rectangle path
        path = QPainterPath()
        path.addRoundedRect(x, y, w, h, self.corner_radius, self.corner_radius)
        
        # Draw border (thicker, more visible)
        pen = QPen(color, self.line_width)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        
        # Draw inner glow (subtle)
        inner_pen = QPen(QColor(color.red(), color.green(), color.blue(), 100), 1)
        painter.setPen(inner_pen)
        inner_path = QPainterPath()
        inner_path.addRoundedRect(
            x + 2, y + 2, w - 4, h - 4,
            self.corner_radius - 2, self.corner_radius - 2
        )
        painter.drawPath(inner_path)
    
    def draw_multiple_tracks(
        self,
        painter: QPainter,
        tracks: List[Dict],
        frame_w: int,
        frame_h: int,
        display_x: int,
        display_y: int,
        display_w: int,
        display_h: int
    ):
        """
        Draw multiple tracks with smart overlay
        
        Args:
            painter: QPainter instance
            tracks: List of track data dicts
            frame_w, frame_h: Original frame dimensions
            display_x, display_y: Display offset
            display_w, display_h: Display dimensions
        """
        if not tracks:
            return
        
        # Calculate scale factors
        scale_x = display_w / frame_w
        scale_y = display_h / frame_h
        
        for track in tracks:
            bbox = track.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            x1, y1, w_box, h_box = bbox
            
            # Scale coordinates
            x_scaled = int(display_x + x1 * scale_x)
            y_scaled = int(display_y + y1 * scale_y)
            w_scaled = int(w_box * scale_x)
            h_scaled = int(h_box * scale_y)
            
            # Draw overlay
            self.draw_track_overlay(
                painter, x_scaled, y_scaled, w_scaled, h_scaled,
                track, scale_x, scale_y
            )

