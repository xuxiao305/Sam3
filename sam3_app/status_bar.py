"""
SAM3 Segment Tool - Status Bar

Bottom status bar showing model state, timing, and operation feedback.
"""

from datetime import datetime
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer


class StatusBar(QWidget):
    """Bottom status bar with auto-clearing messages."""

    def __init__(self):
        super().__init__()
        self._build_ui()
        self.setMaximumHeight(28)

        # Auto-clear timer
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._clear_message)

    def _build_ui(self):
        self.setStyleSheet("background-color: #007acc; border-top: 1px solid #005a9e;")

        layout = QHBoxLayout()
        layout.setContentsMargins(12, 2, 12, 2)

        self.message_label = QLabel("就绪")
        self.message_label.setStyleSheet("color: white; font-size: 11px;")
        layout.addWidget(self.message_label)

        layout.addStretch()

        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: rgba(255,255,255,0.8); font-size: 11px;")
        layout.addWidget(self.stats_label)

        self.setLayout(layout)

    def show_message(self, text: str, duration_ms: int = 3000):
        """Show a message that auto-clears after duration_ms."""
        self.message_label.setText(text)
        if duration_ms > 0:
            self._timer.start(duration_ms)

    def show_error(self, text: str):
        self.setStyleSheet("background-color: #d44e4e; border-top: 1px solid #a33a3a;")
        self.message_label.setText(f"❌ {text}")
        self._timer.start(5000)
        # Reset color after timer
        QTimer.singleShot(5000, self._reset_color)

    def show_success(self, text: str):
        self.setStyleSheet("background-color: #07a77d; border-top: 1px solid #058a68;")
        self.message_label.setText(f"✅ {text}")
        self._timer.start(3000)
        QTimer.singleShot(3000, self._reset_color)

    def set_stats(self, regions: int = 0, confidence: float = 0.0, time_s: float = 0.0):
        parts = []
        if regions > 0:
            parts.append(f"区域: {regions}")
        if confidence > 0:
            parts.append(f"置信度: {confidence:.1%}")
        if time_s > 0:
            parts.append(f"耗时: {time_s:.2f}s")
        self.stats_label.setText("  |  ".join(parts))

    def _clear_message(self):
        self.message_label.setText("就绪")

    def _reset_color(self):
        self.setStyleSheet("background-color: #007acc; border-top: 1px solid #005a9e;")
