"""
Live Charts với PyQtGraph
Tuần 5: Real-time Visualization
Real-time emotion distribution và customer flow charts
"""

import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from qfluentwidgets import HeaderCardWidget
from typing import Dict, List
import numpy as np
from collections import deque


class EmotionDistributionChart(HeaderCardWidget):
    """
    Real-time Emotion Distribution Chart
    Pie chart hoặc bar chart showing emotion distribution
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Emotion Distribution")
        
        # Create PyQtGraph widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('transparent')
        self.plot_widget.setLabel('left', 'Count')
        self.plot_widget.setLabel('bottom', 'Emotion')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Emotion colors
        self.emotion_colors = {
            'happy': '#107C10',      # Green
            'sad': '#0078D7',       # Blue
            'angry': '#E81123',     # Red
            'surprise': '#FFB900',  # Orange
            'fear': '#767676',      # Gray
            'neutral': '#C8C8C8'    # Light Gray
        }
        
        # Data storage
        self.emotion_counts = {
            'happy': 0,
            'sad': 0,
            'angry': 0,
            'surprise': 0,
            'fear': 0,
            'neutral': 0
        }
        
        # Bar chart item
        self.bar_chart = None
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(layout)
    
    def update_emotion(self, emotion: str):
        """Update emotion count"""
        emotion_lower = emotion.lower()
        if emotion_lower in self.emotion_counts:
            self.emotion_counts[emotion_lower] += 1
            self._update_chart()
    
    def update_counts(self, counts: Dict[str, int]):
        """Update emotion counts from dict"""
        for emotion, count in counts.items():
            emotion_lower = emotion.lower()
            if emotion_lower in self.emotion_counts:
                self.emotion_counts[emotion_lower] = count
        self._update_chart()
    
    def _update_chart(self):
        """Update bar chart"""
        # Clear previous
        if self.bar_chart is not None:
            self.plot_widget.removeItem(self.bar_chart)
        
        # Prepare data
        emotions = list(self.emotion_counts.keys())
        counts = [self.emotion_counts[e] for e in emotions]
        colors = [self.emotion_colors.get(e, '#C8C8C8') for e in emotions]
        
        # Create bar chart
        bg = pg.BarGraphItem(
            x=range(len(emotions)),
            height=counts,
            width=0.6,
            brushes=colors,
            pen=pg.mkPen(color='w', width=1)
        )
        
        self.plot_widget.addItem(bg)
        self.bar_chart = bg
        
        # Set x-axis labels
        x_axis = self.plot_widget.getAxis('bottom')
        x_axis.setTicks([[(i, e.capitalize()) for i, e in enumerate(emotions)]])


class CustomerFlowChart(HeaderCardWidget):
    """
    Real-time Customer Flow Chart
    Line chart showing customer count over time
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Customer Flow")
        
        # Create PyQtGraph widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('transparent')
        self.plot_widget.setLabel('left', 'Customers')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(0, 100, padding=0)
        
        # Data storage (time series)
        self.max_points = 100
        self.time_data = deque(maxlen=self.max_points)
        self.customer_data = deque(maxlen=self.max_points)
        self.current_time = 0
        
        # Line chart item
        self.line_plot = None
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(layout)
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_chart)
        self.update_timer.start(1000)  # Update every second
    
    def update_customer_count(self, count: int):
        """Update customer count"""
        self.current_time += 1
        self.time_data.append(self.current_time)
        self.customer_data.append(count)
        self._update_chart()
    
    def _update_chart(self):
        """Update line chart"""
        if len(self.time_data) < 2:
            return
        
        # Clear previous
        if self.line_plot is not None:
            self.plot_widget.removeItem(self.line_plot)
        
        # Convert to numpy arrays
        x = np.array(self.time_data)
        y = np.array(self.customer_data)
        
        # Create line plot
        pen = pg.mkPen(color='#0078D7', width=2)
        self.line_plot = self.plot_widget.plot(x, y, pen=pen, symbol='o', symbolSize=4)
        
        # Auto-scale
        if len(x) > 0:
            self.plot_widget.setXRange(max(0, x[-1] - 50), x[-1] + 5, padding=0)


class FPSChart(HeaderCardWidget):
    """
    Real-time FPS Chart
    Line chart showing FPS over time
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("FPS Performance")
        
        # Create PyQtGraph widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('transparent')
        self.plot_widget.setLabel('left', 'FPS')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(0, 60, padding=0)
        
        # Data storage
        self.max_points = 100
        self.time_data = deque(maxlen=self.max_points)
        self.fps_data = deque(maxlen=self.max_points)
        self.current_time = 0
        
        # Line chart item
        self.line_plot = None
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(layout)
    
    def update_fps(self, fps: float):
        """Update FPS value"""
        self.current_time += 1
        self.time_data.append(self.current_time)
        self.fps_data.append(fps)
        self._update_chart()
    
    def _update_chart(self):
        """Update line chart"""
        if len(self.time_data) < 2:
            return
        
        # Clear previous
        if self.line_plot is not None:
            self.plot_widget.removeItem(self.line_plot)
        
        # Convert to numpy arrays
        x = np.array(self.time_data)
        y = np.array(self.fps_data)
        
        # Color based on FPS (green if >= 15, orange if < 15)
        color = '#107C10' if (y[-1] >= 15 if len(y) > 0 else True) else '#FFB900'
        pen = pg.mkPen(color=color, width=2)
        
        self.line_plot = self.plot_widget.plot(x, y, pen=pen, symbol='o', symbolSize=3)
        
        # Auto-scale
        if len(x) > 0:
            self.plot_widget.setXRange(max(0, x[-1] - 50), x[-1] + 5, padding=0)






