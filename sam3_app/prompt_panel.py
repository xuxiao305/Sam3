"""
SAM3 Segment Tool - Prompt Management Panel

Bottom panel for managing multi-region prompts with tab bar,
matching ComfyUI-SAM3's sam3_multiregion_widget.js tab UI.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QTabBar, QTabWidget, QSizePolicy
)
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtCore import Qt, pyqtSignal

from .prompt_manager import PromptManager, PROMPT_COLORS


class PromptTabBar(QTabBar):
    """Custom tab bar with colored indicators for each prompt region."""

    def __init__(self):
        super().__init__()
        self.setDrawBase(False)
        self.setExpanding(False)
        self.setTabsClosable(True)
        self.setMovable(False)
        self.setStyleSheet("""
            QTabBar::tab {
                background: #2d2d2d;
                color: #ccc;
                border: 1px solid #444;
                padding: 6px 14px;
                margin-right: 2px;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                background: #1a2332;
                border-bottom: 2px solid #0e639c;
            }
            QTabBar::tab:hover {
                background: #363636;
            }
        """)


class PromptPanel(QWidget):
    """Panel for managing prompt regions with tab bar."""

    prompt_changed = pyqtSignal()       # Prompt data changed
    run_requested = pyqtSignal()        # User clicked Run
    clear_active = pyqtSignal()         # Clear active prompt
    clear_all = pyqtSignal()            # Clear all prompts

    def __init__(self, prompt_manager: PromptManager):
        super().__init__()
        self.pm = prompt_manager
        self._build_ui()
        self._refresh_tabs()

    def _build_ui(self):
        self.setStyleSheet("background-color: #252526; border-top: 1px solid #3d3d3d;")

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # Header row
        header = QHBoxLayout()
        title = QLabel("提示区域")
        title.setStyleSheet("color: #aaa; font-weight: 600; font-size: 12px;")
        header.addWidget(title)
        header.addStretch()

        # Help hint
        hint = QLabel("左键=正向点 | 右键=负向点 | Shift+拖拽=框")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        header.addWidget(hint)
        layout.addLayout(header)

        # Tab bar for prompt regions
        self.tab_bar = PromptTabBar()
        self.tab_bar.currentChanged.connect(self._on_tab_changed)
        self.tab_bar.tabCloseRequested.connect(self._on_tab_close)
        layout.addWidget(self.tab_bar)

        # Info row
        info_row = QHBoxLayout()

        self.info_label = QLabel("区域 1 | 正向: 0  负向: 0")
        self.info_label.setStyleSheet("color: #888; font-size: 11px;")
        info_row.addWidget(self.info_label)

        info_row.addStretch()

        # Buttons
        add_btn = QPushButton("+ 添加区域")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #07a77d;
                border: 1px dashed #07a77d;
                border-radius: 3px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #3a3a3a; }
            QPushButton:disabled { color: #555; border-color: #555; }
        """)
        add_btn.clicked.connect(self._on_add_region)
        info_row.addWidget(add_btn)

        clear_btn = QPushButton("清除当前")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #d44e4e;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #3a3a3a; }
        """)
        clear_btn.clicked.connect(self._on_clear_active)
        info_row.addWidget(clear_btn)

        clear_all_btn = QPushButton("清除全部")
        clear_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #d44e4e;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #3a3a3a; }
        """)
        clear_all_btn.clicked.connect(self._on_clear_all)
        info_row.addWidget(clear_all_btn)

        layout.addLayout(info_row)
        self.setLayout(layout)

        self.setMaximumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def _refresh_tabs(self):
        """Rebuild tabs from prompt manager state."""
        self.tab_bar.blockSignals(True)
        while self.tab_bar.count() > 0:
            self.tab_bar.removeTab(0)

        for region in self.pm.regions:
            color = QColor(region.color)
            idx = self.tab_bar.addTab(f"区域 {region.id + 1}")
            self.tab_bar.setTabTextColor(idx, color)

        if 0 <= self.pm.active_index < self.tab_bar.count():
            self.tab_bar.setCurrentIndex(self.pm.active_index)

        self.tab_bar.blockSignals(False)
        self._update_info()

    def _update_info(self):
        """Update the info label."""
        region = self.pm.active
        pos = len(region.positive_points) + len(region.positive_boxes)
        neg = len(region.negative_points) + len(region.negative_boxes)
        self.info_label.setText(
            f"区域 {region.id + 1} | 正向: {pos}  负向: {neg}  | 总计: {self.pm.total_points()} 提示"
        )

    def refresh(self):
        """Refresh tab bar and info."""
        self._refresh_tabs()
        self._update_info()

    def _on_tab_changed(self, index):
        self.pm.set_active(index)
        self._update_info()
        self.prompt_changed.emit()

    def _on_tab_close(self, index):
        self.pm.remove_region(index)
        self._refresh_tabs()
        self.prompt_changed.emit()

    def _on_add_region(self):
        result = self.pm.add_region()
        if result is None:
            return  # Max reached
        self._refresh_tabs()
        self.prompt_changed.emit()

    def _on_clear_active(self):
        self.pm.clear_active()
        self._update_info()
        self.clear_active.emit()

    def _on_clear_all(self):
        self.pm.clear_all()
        self._update_info()
        self.clear_all.emit()
