"""
SAM3 Segment Tool - Interactive Image Canvas

Canvas widget that displays an image and supports:
- Left-click: Add positive point (green circle)
- Right-click: Add negative point (red X)
- Shift+Left-drag: Draw positive bbox (cyan rectangle)
- Shift+Right-drag: Draw negative bbox (red rectangle)
- Right-click on point/box: Delete it
- Scroll wheel: Zoom (future)
- Middle-drag: Pan (future)

Mirrors ComfyUI-SAM3's sam3_multiregion_widget.js functionality.
"""

import numpy as np
from PyQt6.QtWidgets import QFrame
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush, QPolygonF
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal, QPointF

from .prompt_manager import PromptManager


class ImageCanvas(QFrame):
    """Interactive image canvas with point/box annotation support."""

    point_added = pyqtSignal(bool)      # True=positive, False=negative
    box_added = pyqtSignal(bool)        # True=positive, False=negative
    prompt_changed = pyqtSignal()       # Any prompt data changed

    def __init__(self):
        super().__init__()
        self.image: np.ndarray | None = None       # Original image (H, W, 3) RGB
        self.scaled_image: np.ndarray | None = None # Display-sized image
        self.scale_factor: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.prompt_manager: PromptManager | None = None

        # Drawing state
        self._drawing_box = False
        self._box_start: QPoint | None = None
        self._box_current: QPoint | None = None
        self._box_is_positive = True

        # Mask overlay
        self._mask_overlay: np.ndarray | None = None  # (H, W, 4) RGBA

        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")

    def set_prompt_manager(self, manager: PromptManager):
        self.prompt_manager = manager

    def load_image(self, image: np.ndarray):
        """Load an RGB image (numpy array H,W,3 uint8)."""
        self.image = image.copy()
        self._mask_overlay = None
        self._fit_to_canvas()
        self.update()

    def set_mask_overlay(self, overlay: np.ndarray | None):
        """Set mask overlay image (H, W, 4) RGBA uint8. None to clear."""
        self._mask_overlay = overlay
        self.update()

    def _fit_to_canvas(self):
        """Scale image to fit canvas while maintaining aspect ratio."""
        if self.image is None:
            return

        canvas_w = self.width()
        canvas_h = self.height()
        img_h, img_w = self.image.shape[:2]

        scale_w = canvas_w / img_w if img_w > 0 else 1
        scale_h = canvas_h / img_h if img_h > 0 else 1
        self.scale_factor = min(scale_w, scale_h, 1.0)

        new_w = max(1, int(img_w * self.scale_factor))
        new_h = max(1, int(img_h * self.scale_factor))

        import cv2
        self.scaled_image = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2

    def canvas_to_image(self, pos: QPoint) -> tuple[int, int] | None:
        """Convert canvas coordinates to image pixel coordinates. Returns None if outside image."""
        if self.image is None:
            return None
        img_x = (pos.x() - self.offset_x) / self.scale_factor
        img_y = (pos.y() - self.offset_y) / self.scale_factor
        img_h, img_w = self.image.shape[:2]
        ix, iy = int(img_x), int(img_y)
        if 0 <= ix < img_w and 0 <= iy < img_h:
            return ix, iy
        return None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image is not None:
            self._fit_to_canvas()
            self.update()

    # ─── Drawing ───────────────────────────────────────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Background
        painter.fillRect(self.rect(), QColor('#1a1a1a'))

        if self.scaled_image is None:
            painter.setPen(QColor('#666'))
            painter.setFont(QFont('Arial', 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "拖拽或点击上传图像")
            return

        # Draw image
        h, w = self.scaled_image.shape[:2]
        img_rgb = self.scaled_image.copy()  # Already RGB
        bytes_per_line = 3 * w
        q_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        painter.drawImage(QPoint(self.offset_x, self.offset_y), q_image)

        # Draw mask overlay
        if self._mask_overlay is not None:
            self._draw_mask_overlay(painter)

        # Draw prompt annotations
        if self.prompt_manager:
            self._draw_prompts(painter)

        # Draw in-progress box
        if self._drawing_box and self._box_start and self._box_current:
            self._draw_live_box(painter)

        # Draw image dimensions
        if self.image is not None:
            img_h, img_w = self.image.shape[:2]
            painter.setPen(QColor('#888'))
            painter.setFont(QFont('Consolas', 9))
            painter.drawText(10, self.height() - 8, f"{img_w}×{img_h}")

    def _draw_mask_overlay(self, painter: QPainter):
        """Draw semi-transparent mask overlay on the image."""
        import cv2
        overlay = self._mask_overlay
        h, w = overlay.shape[:2]
        new_w = max(1, int(w * self.scale_factor))
        new_h = max(1, int(h * self.scale_factor))
        scaled = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

        q_image = QImage(scaled.data, new_w, new_h, 4 * new_w, QImage.Format.Format_RGBA8888)
        painter.drawImage(QPoint(self.offset_x, self.offset_y), q_image)

    def _draw_prompts(self, painter: QPainter):
        """Draw all prompt points and boxes."""
        for region in self.prompt_manager.regions:
            color = QColor(region.color)
            is_active = (region.id == self.prompt_manager.active_index)

            # Draw positive points (filled circles with +)
            for (px, py) in region.positive_points:
                cx = self.offset_x + px * self.scale_factor
                cy = self.offset_y + py * self.scale_factor
                r = 7 if is_active else 5

                # White outline + colored fill
                painter.setPen(QPen(QColor('white'), 2))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(cx, cy), r, r)

                # Plus sign inside
                painter.setPen(QPen(QColor('white'), 2))
                painter.drawLine(int(cx - 3), int(cy), int(cx + 3), int(cy))
                painter.drawLine(int(cx), int(cy - 3), int(cx), int(cy + 3))

            # Draw negative points (filled circles with ×)
            for (px, py) in region.negative_points:
                cx = self.offset_x + px * self.scale_factor
                cy = self.offset_y + py * self.scale_factor
                r = 7 if is_active else 5

                painter.setPen(QPen(QColor('white'), 2))
                painter.setBrush(QBrush(QColor('#ff3333')))
                painter.drawEllipse(QPointF(cx, cy), r, r)

                # X sign inside
                painter.setPen(QPen(QColor('white'), 2))
                painter.drawLine(int(cx - 3), int(cy - 3), int(cx + 3), int(cy + 3))
                painter.drawLine(int(cx + 3), int(cy - 3), int(cx - 3), int(cy + 3))

            # Draw positive boxes (solid colored rectangles)
            for (x1, y1, x2, y2) in region.positive_boxes:
                cx1 = self.offset_x + x1 * self.scale_factor
                cy1 = self.offset_y + y1 * self.scale_factor
                cx2 = self.offset_x + x2 * self.scale_factor
                cy2 = self.offset_y + y2 * self.scale_factor

                pen_width = 3 if is_active else 2
                painter.setPen(QPen(color, pen_width))
                painter.setBrush(QBrush(color))
                painter.setOpacity(0.15)
                painter.drawRect(QRect(int(cx1), int(cy1), int(cx2 - cx1), int(cy2 - cy1)))
                painter.setOpacity(1.0)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRect(int(cx1), int(cy1), int(cx2 - cx1), int(cy2 - cy1)))

            # Draw negative boxes (dashed red rectangles)
            for (x1, y1, x2, y2) in region.negative_boxes:
                cx1 = self.offset_x + x1 * self.scale_factor
                cy1 = self.offset_y + y1 * self.scale_factor
                cx2 = self.offset_x + x2 * self.scale_factor
                cy2 = self.offset_y + y2 * self.scale_factor

                painter.setPen(QPen(QColor('#ff3333'), 2, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRect(int(cx1), int(cy1), int(cx2 - cx1), int(cy2 - cy1)))

    def _draw_live_box(self, painter: QPainter):
        """Draw the box currently being drawn by the user."""
        color = QColor(self.prompt_manager.active.color) if self._box_is_positive else QColor('#ff3333')
        pen = QPen(color, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        x1, y1 = self._box_start.x(), self._box_start.y()
        x2, y2 = self._box_current.x(), self._box_current.y()
        painter.drawRect(QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))

    # ─── Mouse Events ──────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self.image is None or self.prompt_manager is None:
            return

        pos = event.pos()

        # Shift + click = draw box
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._drawing_box = True
            self._box_start = pos
            self._box_current = pos
            self._box_is_positive = (event.button() == Qt.MouseButton.LeftButton)
            return

        # Right-click on existing point/box = delete
        if event.button() == Qt.MouseButton.RightButton:
            img_pos = self.canvas_to_image(pos)
            if img_pos:
                if self.prompt_manager.remove_point_at(self.prompt_manager.active_index, *img_pos):
                    self.prompt_changed.emit()
                    self.update()
                    return
                if self.prompt_manager.remove_box_at(self.prompt_manager.active_index, *img_pos):
                    self.prompt_changed.emit()
                    self.update()
                    return

        # Normal click = add point
        img_pos = self.canvas_to_image(pos)
        if img_pos is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.prompt_manager.add_positive_point(*img_pos)
            self.point_added.emit(True)
        elif event.button() == Qt.MouseButton.RightButton:
            self.prompt_manager.add_negative_point(*img_pos)
            self.point_added.emit(False)

        self.prompt_changed.emit()
        self.update()

    def mouseMoveEvent(self, event):
        if self._drawing_box:
            self._box_current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drawing_box and self._box_start and self._box_current:
            # Convert canvas coords to image coords
            start = self.canvas_to_image(self._box_start)
            end = self.canvas_to_image(self._box_current)
            if start and end:
                x1, y1 = start
                x2, y2 = end
                # Ensure x1 < x2, y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:  # Minimum box size
                    if self._box_is_positive:
                        self.prompt_manager.add_positive_box(x1, y1, x2, y2)
                    else:
                        self.prompt_manager.add_negative_box(x1, y1, x2, y2)
                    self.box_added.emit(self._box_is_positive)
                    self.prompt_changed.emit()

        self._drawing_box = False
        self._box_start = None
        self._box_current = None
        self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')):
                self.parent()  # Will be handled by main window
