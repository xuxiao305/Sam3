"""
SAM3 Segment Tool - Result Preview Canvas

Displays segmentation visualization results with mask overlay.
"""

import numpy as np
from PyQt6.QtWidgets import QFrame
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt, QPoint


class PreviewCanvas(QFrame):
    """Canvas for displaying segmentation results."""

    def __init__(self):
        super().__init__()
        self.preview_image: np.ndarray | None = None
        self.scale_factor: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")

    def set_result(self, image: np.ndarray):
        """Set preview image (H, W, 3) RGB uint8."""
        self.preview_image = image.copy()
        self._fit_to_canvas()
        self.update()

    def clear(self):
        self.preview_image = None
        self.update()

    def _fit_to_canvas(self):
        if self.preview_image is None:
            return
        canvas_w = self.width()
        canvas_h = self.height()
        img_h, img_w = self.preview_image.shape[:2]
        scale_w = canvas_w / img_w if img_w > 0 else 1
        scale_h = canvas_h / img_h if img_h > 0 else 1
        self.scale_factor = min(scale_w, scale_h, 1.0)
        new_w = max(1, int(img_w * self.scale_factor))
        new_h = max(1, int(img_h * self.scale_factor))
        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.preview_image is not None:
            self._fit_to_canvas()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        painter.fillRect(self.rect(), QColor('#1a1a1a'))

        if self.preview_image is None:
            painter.setPen(QColor('#666'))
            from PyQt6.QtGui import QFont
            painter.setFont(QFont('Arial', 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "等待分割结果...")
            return

        import cv2
        h, w = self.preview_image.shape[:2]
        new_w = max(1, int(w * self.scale_factor))
        new_h = max(1, int(h * self.scale_factor))
        scaled = cv2.resize(self.preview_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        bytes_per_line = 3 * new_w
        q_image = QImage(scaled.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
        painter.drawImage(QPoint(self.offset_x, self.offset_y), q_image)
