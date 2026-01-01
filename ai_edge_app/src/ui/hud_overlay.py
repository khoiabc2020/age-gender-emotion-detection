"""
Dashboard HUD Overlay Widget
Dashboard HUD
Real-time stats overlay trên video
"""

from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QFont, QPen
from qfluentwidgets import CardWidget
from typing import Dict, Optional
import time

class HUDOverlay(CardWidget):
    """
    HUD Overlay Widget
    Hiển thị thông tin real-time trên video
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("hudOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        
        # Stats data
        self.fps: float = 0.0
        self.active_tracks: int = 0
        self.total_customers: int = 0
        self.current_emotion: Optional[str] = None
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_times = 30
        
        # Font
        self.font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        self.small_font = QFont("Segoe UI", 9)
        
        # Colors
        self.text_color = QColor(255, 255, 255)
        self.accent_color = QColor(0, 162, 232)  # Windows 11 blue
        self.warning_color = QColor(255, 185, 0)  # Orange
        self.success_color = QColor(16, 124, 16)  # Green
        
    def update_fps(self, fps: float):
        """Update FPS value"""
        self.fps = fps
        self.frame_times.append(fps)
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
        self.update()
    
    def update_stats(self, stats: Dict):
        """Update HUD stats"""
        self.active_tracks = stats.get('active_tracks', 0)
        self.total_customers = stats.get('total_customers', 0)
        self.current_emotion = stats.get('current_emotion', None)
        self.update()
    
    def paintEvent(self, event):
        """Custom paint event for HUD"""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Position
        x = 20
        y = 20
        line_height = 28
        spacing = 8
        
        # Draw FPS
        fps_color = self.success_color if self.fps >= 15 else self.warning_color
        painter.setPen(QPen(fps_color, 2))
        painter.setFont(self.font)
        painter.drawText(x, y, f"FPS: {self.fps:.1f}")
        
        # Draw Active Tracks
        y += line_height + spacing
        painter.setPen(QPen(self.text_color, 2))
        painter.setFont(self.small_font)
        painter.drawText(x, y, f"Active: {self.active_tracks}")
        
        # Draw Total Customers
        y += line_height
        painter.drawText(x, y, f"Total: {self.total_customers}")
        
        # Draw Current Emotion
        if self.current_emotion:
            y += line_height + spacing
            emotion_colors = {
                'happy': QColor(16, 124, 16),
                'sad': QColor(0, 120, 215),
                'angry': QColor(232, 17, 35),
                'surprise': QColor(255, 185, 0),
                'fear': QColor(118, 118, 118),
                'neutral': QColor(200, 200, 200)
            }
            emotion_color = emotion_colors.get(self.current_emotion.lower(), self.text_color)
            painter.setPen(QPen(emotion_color, 2))
            painter.drawText(x, y, f"Emotion: {self.current_emotion.capitalize()}")
        
        # Draw FPS graph (mini)
        if len(self.frame_times) > 1:
            graph_x = x
            graph_y = y + line_height + spacing
            graph_width = 100
            graph_height = 30
            
            # Background
            painter.setPen(QPen(QColor(255, 255, 255, 50), 1))
            painter.setBrush(QColor(0, 0, 0, 100))
            painter.drawRect(graph_x, graph_y, graph_width, graph_height)
            
            # FPS line
            painter.setPen(QPen(self.accent_color, 2))
            max_fps = max(self.frame_times) if self.frame_times else 30
            min_fps = min(self.frame_times) if self.frame_times else 0
            
            if max_fps > min_fps:
                points = []
                for i, fps_val in enumerate(self.frame_times[-20:]):  # Last 20 frames
                    x_pos = graph_x + (i / 19) * graph_width if len(self.frame_times) > 1 else graph_x
                    normalized = (fps_val - min_fps) / (max_fps - min_fps) if max_fps > min_fps else 0.5
                    y_pos = graph_y + graph_height - (normalized * graph_height)
                    points.append((x_pos, y_pos))
                
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        painter.drawLine(
                            int(points[i][0]), int(points[i][1]),
                            int(points[i+1][0]), int(points[i+1][1])
                        )
        
        painter.end()

