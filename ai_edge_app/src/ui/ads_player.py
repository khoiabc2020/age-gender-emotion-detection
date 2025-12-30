"""
Smart Ads Player với QMediaPlayer
Tuần 6: Dynamic Ads System
Video 4K support với transition effects
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QUrl, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from qfluentwidgets import CardWidget, HeaderCardWidget, BodyLabel
from typing import Dict, List, Optional
from pathlib import Path
import cv2
import numpy as np


class TransitionEffect:
    """Transition effect base class"""
    
    FADE = "fade"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    NONE = "none"
    
    @staticmethod
    def get_all():
        return [
            TransitionEffect.FADE,
            TransitionEffect.SLIDE_LEFT,
            TransitionEffect.SLIDE_RIGHT,
            TransitionEffect.SLIDE_UP,
            TransitionEffect.SLIDE_DOWN,
            TransitionEffect.NONE
        ]


class AdsPlayerWidget(CardWidget):
    """
    Smart Ads Player Widget
    Hỗ trợ video 4K với transition effects
    """
    
    # Signals
    ad_finished = pyqtSignal(str)  # ad_id
    ad_error = pyqtSignal(str, str)  # ad_id, error_message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("adsPlayer")
        self.setMinimumSize(640, 360)
        
        # Media player
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
        # Video widget
        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)
        
        # Fallback image widget (for static ads)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.hide()
        
        # Current ad info
        self.current_ad: Optional[Dict] = None
        self.ad_queue: List[Dict] = []
        self.is_playing = False
        
        # Transition
        self.transition_type = TransitionEffect.FADE
        self.transition_duration = 500  # ms
        self.transition_animation: Optional[QPropertyAnimation] = None
        
        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_widget)
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        
        # Connect signals
        self.media_player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.media_player.errorOccurred.connect(self._on_error)
        self.media_player.playbackStateChanged.connect(self._on_playback_state_changed)
    
    def play_ad(self, ad_data: Dict, transition: str = TransitionEffect.FADE):
        """
        Play advertisement
        
        Args:
            ad_data: Ad data dict với 'ad_id', 'video_path', 'image_path', etc.
            transition: Transition effect type
        """
        self.transition_type = transition
        
        # If currently playing, queue this ad
        if self.is_playing:
            self.ad_queue.append(ad_data)
            return
        
        self.current_ad = ad_data
        self.is_playing = True
        
        # Determine media type
        video_path = ad_data.get('video_path')
        image_path = ad_data.get('image_path')
        
        if video_path and Path(video_path).exists():
            self._play_video(video_path)
        elif image_path and Path(image_path).exists():
            self._play_image(image_path)
        else:
            # Fallback: Show ad name
            self._show_fallback(ad_data.get('name', 'Advertisement'))
    
    def _play_video(self, video_path: str):
        """Play video ad"""
        # Hide image, show video
        self.image_label.hide()
        self.video_widget.show()
        
        # Apply transition
        self._apply_transition_in()
        
        # Load and play video
        url = QUrl.fromLocalFile(video_path)
        self.media_player.setSource(url)
        self.media_player.play()
    
    def _play_image(self, image_path: str):
        """Play static image ad"""
        # Hide video, show image
        self.video_widget.hide()
        self.image_label.show()
        
        # Apply transition
        self._apply_transition_in()
        
        # Load image
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap)
        else:
            self._show_fallback("Image not found")
    
    def _show_fallback(self, text: str):
        """Show fallback text"""
        self.video_widget.hide()
        self.image_label.show()
        
        # Create text image
        label = BodyLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 24px; color: white; background: rgba(0,0,0,0.5);")
        
        # Apply transition
        self._apply_transition_in()
    
    def _apply_transition_in(self):
        """Apply transition effect when starting"""
        if self.transition_type == TransitionEffect.NONE:
            return
        
        # Reset opacity
        self.setWindowOpacity(0.0)
        
        # Create fade animation
        if self.transition_type == TransitionEffect.FADE:
            self.transition_animation = QPropertyAnimation(self, b"windowOpacity")
            self.transition_animation.setDuration(self.transition_duration)
            self.transition_animation.setStartValue(0.0)
            self.transition_animation.setEndValue(1.0)
            self.transition_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self.transition_animation.start()
        
        # Slide animations (simplified - using position)
        elif self.transition_type.startswith("slide_"):
            # For slide, we'll use opacity fade as fallback
            # Full slide implementation would require more complex layout management
            self.transition_animation = QPropertyAnimation(self, b"windowOpacity")
            self.transition_animation.setDuration(self.transition_duration)
            self.transition_animation.setStartValue(0.0)
            self.transition_animation.setEndValue(1.0)
            self.transition_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self.transition_animation.start()
    
    def _apply_transition_out(self, callback):
        """Apply transition effect when ending"""
        if self.transition_type == TransitionEffect.NONE:
            callback()
            return
        
        # Fade out
        if self.transition_animation:
            self.transition_animation.stop()
        
        self.transition_animation = QPropertyAnimation(self, b"windowOpacity")
        self.transition_animation.setDuration(self.transition_duration)
        self.transition_animation.setStartValue(1.0)
        self.transition_animation.setEndValue(0.0)
        self.transition_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.transition_animation.finished.connect(callback)
        self.transition_animation.start()
    
    def _on_media_status_changed(self, status):
        """Handle media status changes"""
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            # Video finished, play next in queue or stop
            self._handle_media_finished()
    
    def _on_playback_state_changed(self, state):
        """Handle playback state changes"""
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self._handle_media_finished()
    
    def _on_error(self, error, error_string):
        """Handle media errors"""
        if self.current_ad:
            ad_id = self.current_ad.get('ad_id', 'unknown')
            self.ad_error.emit(ad_id, error_string)
            self._handle_media_finished()
    
    def _handle_media_finished(self):
        """Handle when media finishes"""
        if self.current_ad:
            ad_id = self.current_ad.get('ad_id', 'unknown')
            self.ad_finished.emit(ad_id)
        
        # Apply transition out
        def on_transition_complete():
            self.is_playing = False
            self.current_ad = None
            
            # Play next in queue
            if self.ad_queue:
                next_ad = self.ad_queue.pop(0)
                self.play_ad(next_ad, self.transition_type)
        
        self._apply_transition_out(on_transition_complete)
    
    def stop(self):
        """Stop current ad"""
        self.media_player.stop()
        self.is_playing = False
        self.current_ad = None
        self.ad_queue.clear()
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)"""
        self.audio_output.setVolume(volume)
    
    def set_transition(self, transition_type: str, duration: int = 500):
        """Set transition effect"""
        self.transition_type = transition_type
        self.transition_duration = duration


class AdsPlayerCard(HeaderCardWidget):
    """
    Ads Player Card với header
    Wrapper cho AdsPlayerWidget
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Advertisement")
        
        # Create player widget
        self.player = AdsPlayerWidget()
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.player)
        layout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(layout)
    
    def play_ad(self, ad_data: Dict, transition: str = TransitionEffect.FADE):
        """Play advertisement"""
        self.player.play_ad(ad_data, transition)
    
    def stop(self):
        """Stop current ad"""
        self.player.stop()
    
    def set_volume(self, volume: float):
        """Set volume"""
        self.player.set_volume(volume)
    
    def set_transition(self, transition_type: str, duration: int = 500):
        """Set transition effect"""
        self.player.set_transition(transition_type, duration)




