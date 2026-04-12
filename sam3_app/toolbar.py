"""
SAM3 Segment Tool - Top Toolbar

Provides controls for: image loading, prompt mode, text prompt,
segmentation trigger, and export.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QComboBox, QFrame, QFileDialog, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont


class Toolbar(QWidget):
    """Top toolbar for the SAM3 application."""

    upload_clicked = pyqtSignal()
    segment_clicked = pyqtSignal()
    export_mask_clicked = pyqtSignal()
    export_json_clicked = pyqtSignal()
    export_vis_clicked = pyqtSignal()
    text_prompt_changed = pyqtSignal(str)
    mode_changed = pyqtSignal(str)   # "point", "text", "video"

    def __init__(self):
        super().__init__()
        self._build_ui()
        self.setMaximumHeight(50)

    def _build_ui(self):
        self.setStyleSheet("background-color: #2d2d2d; border-bottom: 1px solid #3d3d3d;")

        layout = QHBoxLayout()
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(8)

        # Upload image
        self.upload_btn = self._make_button("📁 上传图像", "#0e639c", "#1177bb")
        self.upload_btn.clicked.connect(self.upload_clicked.emit)
        layout.addWidget(self.upload_btn)

        # Separator
        layout.addWidget(self._sep())

        # Mode selector
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["📍 点/框模式", "📝 文本模式", "🎬 视频模式"])
        self.mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #333;
                color: #ddd;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px 10px;
                min-width: 120px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: none; border: none; }
            QComboBox QAbstractItemView {
                background-color: #333;
                color: #ddd;
                selection-background-color: #0e639c;
            }
        """)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)

        # Text prompt input
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("输入文本描述（如：dog, person, car）...")
        self.text_input.setStyleSheet("""
            QLineEdit {
                background-color: #1e1e1e;
                color: #ddd;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px 10px;
                min-width: 200px;
            }
            QLineEdit:focus { border-color: #0e639c; }
        """)
        self.text_input.textChanged.connect(lambda t: self.text_prompt_changed.emit(t))
        self.text_input.setVisible(False)  # Hidden until text mode
        layout.addWidget(self.text_input)

        layout.addWidget(self._sep())

        # Model status
        self.model_status = QLabel("⏳ 模型未加载")
        self.model_status.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.model_status)

        # Image info
        self.image_info = QLabel("")
        self.image_info.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.image_info)

        layout.addStretch()

        # Export buttons
        self.export_mask_btn = self._make_button("💾 导出掩码", "#555", "#666", small=True)
        self.export_mask_btn.clicked.connect(self.export_mask_clicked.emit)
        self.export_mask_btn.setEnabled(False)
        layout.addWidget(self.export_mask_btn)

        self.export_json_btn = self._make_button("📋 导出JSON", "#555", "#666", small=True)
        self.export_json_btn.clicked.connect(self.export_json_clicked.emit)
        self.export_json_btn.setEnabled(False)
        layout.addWidget(self.export_json_btn)

        self.export_vis_btn = self._make_button("🖼️ 导出可视化", "#555", "#666", small=True)
        self.export_vis_btn.clicked.connect(self.export_vis_clicked.emit)
        self.export_vis_btn.setEnabled(False)
        layout.addWidget(self.export_vis_btn)

        layout.addWidget(self._sep())

        # Run button
        self.segment_btn = self._make_button("🚀 开始切割", "#07a77d", "#0d9b6f")
        self.segment_btn.setEnabled(False)
        self.segment_btn.setMinimumWidth(120)
        self.segment_btn.clicked.connect(self.segment_clicked.emit)
        layout.addWidget(self.segment_btn)

        self.setLayout(layout)

    def _make_button(self, text: str, bg: str, hover: str, small: bool = False) -> QPushButton:
        btn = QPushButton(text)
        font_size = "11px" if small else "12px"
        padding = "4px 8px" if small else "6px 14px"
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                color: white;
                border: none;
                border-radius: 3px;
                padding: {padding};
                font-weight: 500;
                font-size: {font_size};
            }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:disabled {{ background-color: #444; color: #777; }}
        """)
        return btn

    def _sep(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #444;")
        return sep

    def _on_mode_changed(self, index):
        modes = ["point", "text", "video"]
        mode = modes[index] if 0 <= index < len(modes) else "point"
        self.text_input.setVisible(mode == "text")
        self.mode_changed.emit(mode)

    def set_model_status(self, loaded: bool, detail: str = ""):
        if loaded:
            self.model_status.setText(f"✅ 模型已加载 {detail}")
            self.model_status.setStyleSheet("color: #07a77d; font-size: 11px;")
        else:
            self.model_status.setText(f"⏳ 模型未加载 {detail}")
            self.model_status.setStyleSheet("color: #888; font-size: 11px;")

    def set_image_info(self, width: int, height: int):
        self.image_info.setText(f"{width}×{height}")

    def set_segmenting(self, active: bool):
        if active:
            self.segment_btn.setText("⏳ 处理中...")
            self.segment_btn.setEnabled(False)
        else:
            self.segment_btn.setText("🚀 开始切割")
            self.segment_btn.setEnabled(True)

    def set_has_result(self, has_result: bool):
        self.export_mask_btn.setEnabled(has_result)
        self.export_json_btn.setEnabled(has_result)
        self.export_vis_btn.setEnabled(has_result)
