"""
Main Window - Modern UI với PyQt6 + QFluentWidgets
Giai đoạn 2 Setup UI Framework
Glassmorphism, Dashboard HUD, Real-time visualization
"""

from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from qfluentwidgets import (
    FluentWindow, NavigationItemPosition, 
    isDarkTheme, setTheme, Theme,
    CardWidget, HeaderCardWidget,
    PrimaryPushButton, LineEdit,
    InfoBar, InfoBarPosition,
    BodyLabel, TitleLabel, CaptionLabel
)
import cv2
import numpy as np
from typing import Optional, Dict
from pathlib import Path

from .glassmorphism import apply_glassmorphism, GLASSMORPHISM_STYLESHEET
from .hud_overlay import HUDOverlay
from .smart_overlay import SmartOverlay
from .live_charts import EmotionDistributionChart, CustomerFlowChart, FPSChart
from .ads_player import AdsPlayerCard, TransitionEffect

class VideoDisplayWidget(CardWidget):
    """Video display widget với glassmorphism effect"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("videoDisplay")
        self.setMinimumSize(640, 480)
        self.current_frame: Optional[np.ndarray] = None
        self.overlay_data: Dict = {}
        
        # Apply glassmorphism
        apply_glassmorphism(self)
        
    def update_frame(self, frame: np.ndarray, overlay_data: Optional[Dict] = None):
        """Update video frame"""
        self.current_frame = frame
        if overlay_data:
            self.overlay_data = overlay_data
        self.update()
    
    def paintEvent(self, event):
        """Custom paint event với glassmorphism và video rendering"""
        super().paintEvent(event)
        
        if self.current_frame is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget size
        widget_rect = self.rect()
        width = widget_rect.width()
        height = widget_rect.height()
        
        # Convert frame to QImage
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit widget (maintain aspect ratio)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            width, height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Center the image
        x = (width - scaled_pixmap.width()) // 2
        y = (height - scaled_pixmap.height()) // 2
        
        # Draw video frame
        painter.drawPixmap(x, y, scaled_pixmap)
        
        # Draw overlay data (bounding boxes, labels, etc.)
        # Smart Overlay với rounded boxes và emotion colors
        if self.overlay_data:
            self._draw_smart_overlay(painter, x, y, scaled_pixmap.width(), scaled_pixmap.height())
        
        painter.end()
    
    def _draw_smart_overlay(self, painter: QPainter, x: int, y: int, w: int, h: int):
        """Draw smart overlay với rounded boxes và emotion colors ()"""
        if self.current_frame is None:
            return
        
        frame_h, frame_w = self.current_frame.shape[:2]
        
        # Draw tracks với smart overlay
        if 'tracks' in self.overlay_data:
            tracks = self.overlay_data['tracks']
            self.overlay_renderer.draw_multiple_tracks(
                painter, tracks, frame_w, frame_h, x, y, w, h
            )

class StatsCardWidget(HeaderCardWidget):
    """Stats card widget với glassmorphism"""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setTitle(title)
        self.setObjectName("statsCard")
        
        # Apply glassmorphism
        apply_glassmorphism(self)
        
        # Content layout
        self.content_label = BodyLabel("--")
        self.content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #0078D4;")
        
        layout = QVBoxLayout()
        layout.addWidget(self.content_label)
        layout.addStretch()
        self.viewLayout.addLayout(layout)
    
    def update_value(self, value: str):
        """Update displayed value"""
        self.content_label.setText(str(value))

class MainWindow(FluentWindow):
    """
    Main Window - Modern UI
    Glassmorphism, Dashboard HUD, Real-time visualization
    """
    
    # Signals
    video_frame_ready = pyqtSignal(np.ndarray, dict)  # frame, overlay_data
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Retail AI - Ultimate Edition")
        self.resize(1920, 1080)
        
        # Setup theme
        setTheme(Theme.AUTO)
        
        # Apply global glassmorphism stylesheet
        self.setStyleSheet(GLASSMORPHISM_STYLESHEET)
        
        # Stats tracking
        self.stats = {
            'fps': 0.0,
            'active_tracks': 0,
            'total_customers': 0,
            'current_emotion': None
        }
        
        # Initialize UI components
        self._init_ui()
        
        # Video timer
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self._update_video)
        self.video_timer.setInterval(33)  # ~30 FPS
        
        # Stats update timer
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start(100)  # Update every 100ms
        
    def _init_ui(self):
        """Initialize UI components"""
        # Main video display
        self.video_widget = VideoDisplayWidget()
        self.video_widget.setMinimumSize(1280, 720)
        
        # HUD Overlay (on top of video)
        self.hud_overlay = HUDOverlay(self.video_widget)
        self.hud_overlay.setGeometry(0, 0, 300, 200)
        self.hud_overlay.raise_()  # Bring to front
        
        # Stats cards
        self.fps_card = StatsCardWidget("FPS")
        self.fps_card.setFixedHeight(120)
        
        self.customer_card = StatsCardWidget("Customers")
        self.customer_card.setFixedHeight(120)
        
        self.emotion_card = StatsCardWidget("Emotion")
        self.emotion_card.setFixedHeight(120)
        
        # Layout
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Left: Video + FPS
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        left_layout.addWidget(self.video_widget, stretch=3)
        left_layout.addWidget(self.fps_card, stretch=0)
        
        # Right: Stats + Charts
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        right_layout.addWidget(self.customer_card, stretch=0)
        right_layout.addWidget(self.emotion_card, stretch=0)
        
        # Live Charts
        self.emotion_chart = EmotionDistributionChart()
        self.emotion_chart.setFixedHeight(200)
        right_layout.addWidget(self.emotion_chart, stretch=0)
        
        self.customer_flow_chart = CustomerFlowChart()
        self.customer_flow_chart.setFixedHeight(200)
        right_layout.addWidget(self.customer_flow_chart, stretch=0)
        
        self.fps_chart = FPSChart()
        self.fps_chart.setFixedHeight(200)
        right_layout.addWidget(self.fps_chart, stretch=0)
        
        # Dynamic Ads Player
        self.ads_player = AdsPlayerCard()
        self.ads_player.setFixedHeight(400)
        right_layout.addWidget(self.ads_player, stretch=0)
        
        # Connect ads player signals
        self.ads_player.player.ad_finished.connect(self._on_ad_finished)
        self.ads_player.player.ad_error.connect(self._on_ad_error)
        
        right_layout.addStretch()
        
        main_layout.addLayout(left_layout, stretch=2)
        main_layout.addLayout(right_layout, stretch=1)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Connect video frame signal
        self.video_frame_ready.connect(self._on_video_frame)
    
    def _on_video_frame(self, frame: np.ndarray, overlay_data: dict):
        """Handle video frame from processing thread"""
        self.video_widget.update_frame(frame, overlay_data)
    
    def _update_video(self):
        """Update video frame (called by timer)"""
        # This will be connected to video processing thread
        # For now, placeholder
        pass
    
    def _update_stats(self):
        """Update stats display"""
        # Update HUD
        self.hud_overlay.update_fps(self.stats['fps'])
        self.hud_overlay.update_stats(self.stats)
        
        # Update cards
        self.fps_card.update_value(f"{self.stats['fps']:.1f}")
        self.customer_card.update_value(str(self.stats['total_customers']))
        emotion_text = self.stats['current_emotion'] or "N/A"
        self.emotion_card.update_value(emotion_text.capitalize())
        
        # Update live charts
        self.fps_chart.update_fps(self.stats['fps'])
        self.customer_flow_chart.update_customer_count(self.stats['total_customers'])
        
        # Update emotion chart if emotion detected
        if self.stats['current_emotion']:
            self.emotion_chart.update_emotion(self.stats['current_emotion'])
    
    def update_stats(self, stats: Dict):
        """Update application stats"""
        self.stats.update(stats)
    
    def start_video(self):
        """Start video processing"""
        self.video_timer.start()
    
    def stop_video(self):
        """Stop video processing"""
        self.video_timer.stop()
    
    def set_video_frame(self, frame: np.ndarray, overlay_data: Optional[Dict] = None):
        """Set video frame from external source"""
        self.video_frame_ready.emit(frame, overlay_data or {})
    
    def play_advertisement(self, ad_data: Dict, transition: str = TransitionEffect.FADE):
        """
        Play advertisement in ads player
        
        Args:
            ad_data: Ad data dict với 'ad_id', 'video_path', 'image_path', 'name'
            transition: Transition effect (fade, slide_left, slide_right, etc.)
        """
        self.ads_player.play_ad(ad_data, transition)
    
    def _on_ad_finished(self, ad_id: str):
        """Handle when ad finishes"""
        # Can be used for analytics or next ad selection
        pass
    
    def _on_ad_error(self, ad_id: str, error_message: str):
        """Handle ad playback errors"""
        print(f"Ad error ({ad_id}): {error_message}")

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    # Test with dummy frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Test Video", (200, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    window.set_video_frame(test_frame)
    
    # Test stats
    window.update_stats({
        'fps': 30.0,
        'active_tracks': 2,
        'total_customers': 10,
        'current_emotion': 'happy'
    })
    
    sys.exit(app.exec())
