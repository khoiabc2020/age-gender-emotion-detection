"""
Glassmorphism (Acrylic) Effect Stylesheet
Tuần 4: Modern UI Framework
Windows 11-style glass effect
"""

GLASSMORPHISM_STYLESHEET = """
/* Glassmorphism Card Widget */
CardWidget {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

CardWidget:hover {
    background-color: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Header Card with Glassmorphism */
HeaderCardWidget {
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
}

/* Video Display with Glassmorphism */
#videoDisplay {
    background-color: rgba(0, 0, 0, 0.3);
    border-radius: 20px;
    border: 2px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

/* Stats Card */
#statsCard {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 0, y2: 1,
        stop: 0 rgba(255, 255, 255, 0.12),
        stop: 1 rgba(255, 255, 255, 0.05)
    );
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

/* HUD Overlay */
#hudOverlay {
    background-color: rgba(0, 0, 0, 0.4);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

/* Dark Theme Adjustments */
[theme="dark"] CardWidget {
    background-color: rgba(30, 30, 30, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

[theme="dark"] HeaderCardWidget {
    background-color: rgba(20, 20, 20, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

[theme="dark"] #videoDisplay {
    background-color: rgba(0, 0, 0, 0.5);
    border: 2px solid rgba(255, 255, 255, 0.05);
}

[theme="dark"] #statsCard {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 0, y2: 1,
        stop: 0 rgba(40, 40, 40, 0.7),
        stop: 1 rgba(20, 20, 20, 0.5)
    );
    border: 1px solid rgba(255, 255, 255, 0.1);
}
"""


def apply_glassmorphism(widget):
    """
    Apply glassmorphism effect to widget
    
    Args:
        widget: QWidget to apply effect
    """
    widget.setStyleSheet(GLASSMORPHISM_STYLESHEET)






