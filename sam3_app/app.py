"""
SAM3 Segment Tool - Main Application Window

Combines all UI components: toolbar, canvas, preview, prompt panel, status bar.
Orchestrates SAM3 backend for interactive and text-based segmentation.
"""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFileDialog, QMessageBox, QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QAction

from .canvas import ImageCanvas
from .preview import PreviewCanvas
from .prompt_manager import PromptManager, PROMPT_COLORS
from .prompt_panel import PromptPanel
from .toolbar import Toolbar
from .status_bar import StatusBar
from .backend import SAM3Backend
from .export import export_mask_png, export_prompts_json, export_visualization, export_individual_masks

log = logging.getLogger("sam3_app")


# ─── Worker Signals ──────────────────────────────────────────────────────────

class SegmentWorker(QObject):
    """Run segmentation in a background thread."""
    finished = pyqtSignal(list, list, list, np.ndarray, float)  # masks, scores, boxes, vis_image, time_s
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, backend: SAM3Backend, prompts, img_w, img_h, mode, text_prompt=""):
        super().__init__()
        self.backend = backend
        self.prompts = prompts
        self.img_w = img_w
        self.img_h = img_h
        self.mode = mode
        self.text_prompt = text_prompt

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray):
        """Compute axis-aligned bounding box from a binary mask.
        Returns [x1, y1, x2, y2] in pixel coords, or None if mask is empty."""
        m = mask.squeeze() if mask.ndim > 2 else mask
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)
        if not rows.any():
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return [int(cmin), int(rmin), int(cmax), int(rmax)]

    def run(self):
        try:
            start = time.time()
            boxes = []
            if self.mode == "text":
                masks, scores, boxes, vis = self.backend.segment_text(
                    self.text_prompt, self.img_w, self.img_h,
                )
                # Normalize boxes to pixel coords if they're in normalized format
                if boxes and isinstance(boxes, list) and len(boxes) > 0:
                    if isinstance(boxes[0], list) and len(boxes[0]) == 4:
                        first_val = boxes[0][0]
                        if isinstance(first_val, (int, float)) and 0 <= first_val <= 1:
                            boxes = [
                                [b[0]*self.img_w, b[1]*self.img_h, b[2]*self.img_w, b[3]*self.img_h]
                                for b in boxes
                            ]
            else:
                masks, scores, vis = self.backend.segment_interactive(
                    self.prompts, self.img_w, self.img_h,
                )
                # Compute bounding boxes from masks for interactive mode
                boxes = [self._mask_to_bbox(m) for m in masks]

            elapsed = time.time() - start
            self.finished.emit(masks, scores, boxes, vis, elapsed)
        except Exception as e:
            log.exception("Segmentation failed")
            self.error.emit(str(e))


class LoadWorker(QObject):
    """Load model in a background thread."""
    finished = pyqtSignal(bool)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, backend: SAM3Backend):
        super().__init__()
        self.backend = backend

    def run(self):
        try:
            result = self.backend.load_model(progress_cb=self.progress.emit)
            self.finished.emit(result)
        except Exception as e:
            log.exception("Model loading failed")
            self.error.emit(str(e))


# ─── Main Window ─────────────────────────────────────────────────────────────

class SAM3App(QMainWindow):
    """Main application window for SAM3 Segment Tool."""

    def __init__(
        self,
        initial_image: Optional[str] = None,
        export_dir: Optional[str] = None,
        export_basename: Optional[str] = None,
        auto_exit_on_export: bool = False,
    ):
        super().__init__()
        self.setWindowTitle("SAM3 分割工具")
        self.setGeometry(80, 80, 1500, 900)
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(self._global_stylesheet())

        # Data
        self.prompt_manager = PromptManager()
        self.backend = SAM3Backend()
        self.current_image: Optional[np.ndarray] = None  # (H, W, 3) RGB
        self.current_image_path: Optional[str] = None
        self.result_masks: list = []
        self.result_scores: list = []
        self.result_boxes: list = []  # [[x1,y1,x2,y2], ...] pixel coords
        self.result_vis: Optional[np.ndarray] = None
        self.current_mode = "point"  # "point" or "text"

        # Headless / bridge integration parameters (optional CLI flags).
        # When `export_dir` is set, "导出 JSON" skips the file dialog and writes
        # directly into that directory using `<export_basename>.json` and
        # `<export_basename>_mask.png`. When `auto_exit_on_export` is True, the
        # window closes itself once the export completes — the parent process
        # (Vite dev server) then knows the user finished and reads the files.
        self._export_dir: Optional[str] = export_dir
        self._export_basename: Optional[str] = export_basename or "segmentation"
        self._auto_exit_on_export: bool = bool(auto_exit_on_export)
        self._initial_image: Optional[str] = initial_image

        # Workers
        self._segment_worker = None
        self._segment_thread = None
        self._load_worker = None
        self._load_thread = None

        # Build UI
        self._build_ui()
        self._connect_signals()

        # Load model on startup
        self._start_model_load()

        # If launched with --image, queue an automatic load *after* the event
        # loop starts. We can't call _load_image() right here because the
        # backend hasn't finished loading yet; instead defer to a 0-ms timer
        # that runs once the window is shown.
        if self._initial_image:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._load_image(self._initial_image))

    def _build_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        central.setLayout(root_layout)

        # Toolbar
        self.toolbar = Toolbar()
        root_layout.addWidget(self.toolbar)

        # Loading bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximumHeight(3)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { background: #2d2d2d; border: none; }
            QProgressBar::chunk { background: #0e639c; }
        """)
        self.progress_bar.hide()
        root_layout.addWidget(self.progress_bar)

        # Main content: splitter with canvas + preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: #333; width: 3px; }")

        # Left panel (input + prompt management)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.canvas = ImageCanvas()
        self.canvas.set_prompt_manager(self.prompt_manager)
        left_layout.addWidget(self.canvas, stretch=1)

        self.prompt_panel = PromptPanel(self.prompt_manager)
        left_layout.addWidget(self.prompt_panel)

        left_panel.setLayout(left_layout)

        # Right panel (preview + stats)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.preview = PreviewCanvas()
        right_layout.addWidget(self.preview, stretch=1)

        right_panel.setLayout(right_layout)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([750, 750])
        root_layout.addWidget(splitter, stretch=1)

        # Status bar
        self.status_bar = StatusBar()
        root_layout.addWidget(self.status_bar)

    def _connect_signals(self):
        # Toolbar
        self.toolbar.upload_clicked.connect(self._on_upload)
        self.toolbar.segment_clicked.connect(self._on_segment)
        self.toolbar.mode_changed.connect(self._on_mode_changed)
        self.toolbar.text_prompt_changed.connect(self._on_text_prompt_changed)
        self.toolbar.export_mask_clicked.connect(self._on_export_mask)
        self.toolbar.export_json_clicked.connect(self._on_export_json)
        self.toolbar.export_vis_clicked.connect(self._on_export_vis)

        # Canvas
        self.canvas.point_added.connect(self._on_point_added)
        self.canvas.box_added.connect(self._on_box_added)
        self.canvas.prompt_changed.connect(self._on_prompt_changed)

        # Prompt panel
        self.prompt_panel.prompt_changed.connect(self._on_prompt_changed)
        self.prompt_panel.clear_active.connect(self._on_clear_active)
        self.prompt_panel.clear_all.connect(self._on_clear_all)

    def _global_stylesheet(self) -> str:
        return """
            QMainWindow { background-color: #1e1e1e; color: #e0e0e0; }
            QWidget { background-color: #1e1e1e; color: #e0e0e0; }
            QSplitter { background-color: #1e1e1e; }
        """

    # ─── Model Loading ──────────────────────────────────────────────────────

    def _start_model_load(self):
        self.progress_bar.show()
        self.toolbar.set_model_status(False, "(加载中...)")

        self._load_thread = QThread()
        self._load_worker = LoadWorker(self.backend)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.progress.connect(self._on_load_progress)
        self._load_worker.finished.connect(self._on_model_loaded)
        self._load_worker.error.connect(self._on_model_load_error)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.error.connect(self._load_thread.quit)
        self._load_thread.start()

    def _on_load_progress(self, text: str):
        self.toolbar.set_model_status(False, f"({text})")

    def _on_model_loaded(self, success: bool):
        self.progress_bar.hide()
        if success:
            size_mb = self.backend.model_size_mb
            self.toolbar.set_model_status(True, f"({size_mb:.0f} MB)")
            self.status_bar.show_success(f"模型加载成功 ({size_mb:.0f} MB)")

            # Bridge mode: an image may have been preloaded before the backend
            # was ready. In that case `self.current_image` is set but the
            # backbone features were skipped — extract them now so the user
            # can immediately point/click without re-uploading.
            if self.current_image is not None:
                try:
                    from PIL import Image as PILImage
                    self.status_bar.show_message("提取图像特征中...", 0)
                    QApplication.processEvents()
                    self.backend.set_image(PILImage.fromarray(self.current_image))
                    self.status_bar.show_success("特征提取完成")
                except Exception as e:
                    log.exception("Deferred feature extraction failed")
                    self.status_bar.show_error(f"特征提取失败: {str(e)[:80]}")
        else:
            self.toolbar.set_model_status(False)
            self.status_bar.show_error("模型加载失败")

    def _on_model_load_error(self, error: str):
        self.progress_bar.hide()
        self.toolbar.set_model_status(False)
        self.status_bar.show_error(f"模型加载失败: {error[:60]}")

    # ─── Image Loading ──────────────────────────────────────────────────────

    def _on_upload(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.webp *.tiff);;所有文件 (*)"
        )
        if not file_path:
            return

        self._load_image(file_path)

    def _load_image(self, file_path: str):
        image = cv2.imread(file_path)
        if image is None:
            self.status_bar.show_error("图像加载失败")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = image
        self.current_image_path = file_path

        h, w = image.shape[:2]
        self.toolbar.set_image_info(w, h)
        self.toolbar.segment_btn.setEnabled(True)

        # Reset prompts
        self.prompt_manager.clear_all()
        self.prompt_panel.refresh()

        # Clear results
        self.result_masks = []
        self.result_scores = []
        self.result_boxes = []
        self.result_vis = None
        self.preview.clear()
        self.canvas.set_mask_overlay(None)
        self.toolbar.set_has_result(False)

        # Display image
        self.canvas.load_image(image)
        self.status_bar.show_success(f"图像已加载: {Path(file_path).name}")

        # Set image in backend (extract backbone features)
        if self.backend.is_loaded:
            self.status_bar.show_message("提取图像特征中...", 0)
            QApplication.processEvents()
            try:
                from PIL import Image as PILImage
                self.backend.set_image(PILImage.fromarray(image))
                self.status_bar.show_success("特征提取完成")
            except Exception as e:
                log.exception("Feature extraction failed")
                self.status_bar.show_error(f"特征提取失败: {str(e)[:80]}")
        else:
            self.status_bar.show_message("模型加载中，请稍后再试...", 3000)

    # ─── Segmentation ───────────────────────────────────────────────────────

    def _on_segment(self):
        if self.current_image is None:
            self.status_bar.show_error("请先上传图像")
            return

        if not self.backend.is_loaded:
            self.status_bar.show_error("模型未加载")
            return

        # Auto-extract features if state is missing
        if self.backend._state is None and self.current_image is not None:
            self.status_bar.show_message("提取图像特征中...", 0)
            QApplication.processEvents()
            try:
                from PIL import Image as PILImage
                self.backend.set_image(PILImage.fromarray(self.current_image))
                self.status_bar.show_success("特征提取完成")
            except Exception as e:
                log.exception("Feature extraction failed during segment")
                self.status_bar.show_error(f"特征提取失败: {str(e)[:80]}")
                return

        if self.current_mode == "point":
            if not self.prompt_manager.has_content():
                self.status_bar.show_error("请标记至少一个提示点或框")
                return
            self._run_point_segment()
        elif self.current_mode == "text":
            text = self.toolbar.text_input.text().strip()
            if not text:
                self.status_bar.show_error("请输入文本描述")
                return
            self._run_text_segment(text)

    def _run_point_segment(self):
        h, w = self.current_image.shape[:2]
        prompts = self.prompt_manager.to_sam3_format(w, h)

        if not prompts:
            self.status_bar.show_error("无有效提示")
            return

        self._start_segmentation(
            mode="point",
            prompts=prompts,
            img_w=w,
            img_h=h,
        )

    def _run_text_segment(self, text: str):
        h, w = self.current_image.shape[:2]
        self._start_segmentation(
            mode="text",
            prompts=None,
            img_w=w,
            img_h=h,
            text_prompt=text,
        )

    def _start_segmentation(self, mode, prompts=None, img_w=0, img_h=0, text_prompt=""):
        self.toolbar.set_segmenting(True)
        self.progress_bar.show()
        self.status_bar.show_message("分割中...", 0)

        self._segment_thread = QThread()
        self._segment_worker = SegmentWorker(
            self.backend, prompts, img_w, img_h, mode, text_prompt
        )
        self._segment_worker.moveToThread(self._segment_thread)
        self._segment_thread.started.connect(self._segment_worker.run)
        self._segment_worker.finished.connect(self._on_segment_done)
        self._segment_worker.error.connect(self._on_segment_error)
        self._segment_worker.finished.connect(self._segment_thread.quit)
        self._segment_worker.error.connect(self._segment_thread.quit)
        self._segment_thread.start()

    def _on_segment_done(self, masks, scores, boxes, vis_image, elapsed):
        self.toolbar.set_segmenting(False)
        self.progress_bar.hide()

        self.result_masks = masks
        self.result_scores = scores
        self.result_boxes = boxes
        self.result_vis = vis_image

        # Update preview
        self.preview.set_result(vis_image)
        self.toolbar.set_has_result(True)

        # Update canvas mask overlay
        try:
            self._update_canvas_overlay()
        except Exception as e:
            log.warning(f"Failed to update canvas overlay: {e}")

        # Update status
        n_regions = len(masks)
        avg_conf = sum(scores) / len(scores) if scores else 0
        self.status_bar.set_stats(regions=n_regions, confidence=avg_conf, time_s=elapsed)
        self.status_bar.show_success(f"分割完成: {n_regions} 个区域 ({elapsed:.2f}s)")

    def _on_segment_error(self, error: str):
        self.toolbar.set_segmenting(False)
        self.progress_bar.hide()
        self.status_bar.show_error(f"分割失败: {error[:60]}")

    def _update_canvas_overlay(self):
        """Create RGBA overlay from result masks and set on canvas."""
        if not self.result_masks or self.current_image is None:
            self.canvas.set_mask_overlay(None)
            return

        h, w = self.current_image.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        for i, mask in enumerate(self.result_masks):
            import cv2
            color_hex = PROMPT_COLORS[i % len(PROMPT_COLORS)]
            r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)

            # Ensure mask is 2D (H, W) — may come as (1, H, W) or (H, W, 1)
            if mask.ndim > 2:
                mask = mask.squeeze()

            # Resize mask if needed
            if mask.shape != (h, w):
                mask_resized = cv2.resize(
                    mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                mask_resized = mask.astype(bool) if mask.dtype != bool else mask

            overlay[mask_resized, 0] = r
            overlay[mask_resized, 1] = g
            overlay[mask_resized, 2] = b
            overlay[mask_resized, 3] = 128  # 50% alpha

        self.canvas.set_mask_overlay(overlay)

    # ─── Prompt Events ──────────────────────────────────────────────────────

    def _on_point_added(self, is_positive: bool):
        self.prompt_panel.refresh()
        kind = "正向" if is_positive else "负向"
        self.status_bar.show_message(f"已添加{kind}点", 1500)

    def _on_box_added(self, is_positive: bool):
        self.prompt_panel.refresh()
        kind = "正向" if is_positive else "负向"
        self.status_bar.show_message(f"已添加{kind}框", 1500)

    def _on_prompt_changed(self):
        self.prompt_panel.refresh()
        self.canvas.update()

    def _on_clear_active(self):
        self.canvas.update()
        self.prompt_panel.refresh()

    def _on_clear_all(self):
        self.canvas.update()
        self.prompt_panel.refresh()

    # ─── Mode Changes ───────────────────────────────────────────────────────

    def _on_mode_changed(self, mode: str):
        self.current_mode = mode
        if mode == "text":
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.canvas.setCursor(Qt.CursorShape.CrossCursor)

    def _on_text_prompt_changed(self, text: str):
        pass  # Text is read at segment time

    # ─── Export ─────────────────────────────────────────────────────────────

    def _on_export_mask(self):
        if not self.result_masks or self.current_image is None:
            self.status_bar.show_error("无分割结果可导出")
            return

        h, w = self.current_image.shape[:2]
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出掩码", "mask.png", "PNG (*.png)"
        )
        if file_path:
            export_mask_png(self.result_masks, file_path, w, h)
            self.status_bar.show_success(f"掩码已导出: {Path(file_path).name}")

    def _on_export_json(self):
        if self.current_image is None:
            self.status_bar.show_error("无数据可导出")
            return

        if not self.result_masks:
            self.status_bar.show_error("请先执行分割再导出")
            return

        try:
            h, w = self.current_image.shape[:2]

            # Headless / bridge mode: skip the save dialog when the parent
            # process gave us a fixed export directory at startup.
            if self._export_dir:
                from pathlib import Path as _Path
                _Path(self._export_dir).mkdir(parents=True, exist_ok=True)
                file_path = str(_Path(self._export_dir) / f"{self._export_basename}.json")
            else:
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "导出JSON", "segmentation.json", "JSON (*.json)"
                )
                if not file_path:
                    return

            json_path = Path(file_path)
            out_dir = json_path.parent
            image_name = Path(self.current_image_path).name if self.current_image_path else ""
            mask_filename = json_path.stem + "_mask.png"
            mask_path = str(out_dir / mask_filename)

            # Export combined mask PNG
            import cv2
            from PIL import Image as PILImage
            n = len(self.result_masks)
            combined = np.zeros((h, w), dtype=np.uint8)
            mask_values = []
            bboxes = []
            for i, mask in enumerate(self.result_masks):
                m = mask.squeeze() if mask.ndim > 2 else mask
                m = m.astype(np.uint8)  # ensure uint8 for cv2
                if m.shape != (h, w):
                    m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    m_bool = m_resized > 0
                else:
                    m_bool = m > 0
                val = 255 if n == 1 else int(255 * (i + 1) / n)
                combined[m_bool] = val
                mask_values.append(val)

                # Compute bbox (xyxy) from mask; None if mask is empty
                ys, xs = np.where(m_bool)
                if xs.size > 0 and ys.size > 0:
                    x_min = int(xs.min())
                    y_min = int(ys.min())
                    x_max = int(xs.max())
                    y_max = int(ys.max())
                    bboxes.append({
                        "xyxy": [x_min, y_min, x_max, y_max],
                        "xywh": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
                    })
                else:
                    bboxes.append(None)

            PILImage.fromarray(combined, mode='L').save(mask_path)

            # Build objects list
            objects = []
            for i in range(n):
                # Label: use text prompt if available, else Region_N
                if self.current_mode == "text":
                    text = self.toolbar.text_input.text().strip()
                    if text:
                        parts = [p.strip() for p in text.split(',')]
                        label = f"Object_{parts[i]}" if i < len(parts) else f"Object_{i+1}"
                    else:
                        label = f"Object_{i+1}"
                else:
                    label = f"Region_{i+1}"
                objects.append({
                    "label": label,
                    "mask_value": mask_values[i],
                    "bbox": bboxes[i],
                })

            export_data = {
                "image": image_name,
                "mask_png": mask_filename,
                "objects": objects,
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            self.status_bar.show_success(f"JSON+Mask已导出: {json_path.name}, {mask_filename}")

            # Bridge mode: parent process is waiting for the export to land,
            # then expects the window to close itself.
            if self._auto_exit_on_export:
                from PyQt6.QtCore import QTimer
                # Small delay so the user sees the "exported" toast briefly.
                QTimer.singleShot(400, self.close)
        except Exception as e:
            log.exception("Export JSON failed")
            self.status_bar.show_error(f"导出失败: {str(e)[:80]}")

    def _on_export_vis(self):
        if self.result_vis is None:
            self.status_bar.show_error("无可视化结果可导出")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出可视化", "segmentation.png", "PNG (*.png)"
        )
        if file_path:
            export_visualization(self.result_vis, file_path)
            self.status_bar.show_success(f"可视化已导出: {Path(file_path).name}")

    # ─── Drag & Drop ────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')):
                self._load_image(path)

    # ─── Cleanup ────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.backend.is_loaded:
            self.backend.unload()
        event.accept()
