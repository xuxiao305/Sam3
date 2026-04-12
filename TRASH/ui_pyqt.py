import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QFrame, QScrollArea,
    QGridLayout
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon
from PyQt6.QtCore import Qt, QSize, QPoint, QRect, pyqtSignal, QTimer
from PyQt6.QtWidgets import QListWidget, QListWidgetItem


class PromptManager:
    """管理提示点数据"""
    
    PROMPT_COLORS = [
        '#ff00ff',  # Magenta - Prompt 1
        '#ffff00',  # Yellow - Prompt 2
        '#00ffff',  # Cyan - Prompt 3
        '#ff6600',  # Orange - Prompt 4
        '#00ff00',  # Green - Prompt 5
        '#ff0099',  # Pink - Prompt 6
    ]
    
    def __init__(self):
        self.prompts: List[Dict] = []
        self.active_index = 0
    
    def add_prompt(self):
        """添加新的提示组"""
        if len(self.prompts) >= 6:
            return False
        
        prompt = {
            'id': len(self.prompts) + 1,
            'positive_points': [],
            'negative_points': [],
            'color': self.PROMPT_COLORS[len(self.prompts)],
        }
        self.prompts.append(prompt)
        self.active_index = len(self.prompts) - 1
        return True
    
    def remove_prompt(self, index: int):
        """删除提示组"""
        if len(self.prompts) <= 1:
            return False
        self.prompts.pop(index)
        if self.active_index >= len(self.prompts):
            self.active_index = len(self.prompts) - 1
        return True
    
    def add_positive_point(self, x: int, y: int):
        """添加正向点"""
        if self.prompts:
            self.prompts[self.active_index]['positive_points'].append([x, y])
    
    def add_negative_point(self, x: int, y: int):
        """添加负向点"""
        if self.prompts:
            self.prompts[self.active_index]['negative_points'].append([x, y])
    
    def clear_current_prompt(self):
        """清除当前提示组的所有点"""
        if self.prompts:
            count = len(self.prompts[self.active_index]['positive_points']) + \
                   len(self.prompts[self.active_index]['negative_points'])
            self.prompts[self.active_index]['positive_points'] = []
            self.prompts[self.active_index]['negative_points'] = []
            return count
        return 0
    
    def clear_all_prompts(self):
        """清除所有提示点"""
        count = 0
        for prompt in self.prompts:
            count += len(prompt['positive_points']) + len(prompt['negative_points'])
            prompt['positive_points'] = []
            prompt['negative_points'] = []
        return count
    
    def set_active_prompt(self, index: int):
        """设置活跃提示组"""
        if 0 <= index < len(self.prompts):
            self.active_index = index
    
    def get_active_prompt(self):
        """获取活跃提示组"""
        if self.prompts:
            return self.prompts[self.active_index]
        return None
    
    def get_total_points(self):
        """获取总提示点数"""
        count = 0
        for prompt in self.prompts:
            count += len(prompt['positive_points']) + len(prompt['negative_points'])
        return count
    
    def has_points(self):
        """检查是否有提示点"""
        return any(p['positive_points'] or p['negative_points'] for p in self.prompts)


class ImageCanvas(QFrame):
    """图像画布，支持点击标记"""
    
    point_added = pyqtSignal(bool)  # True=正向点，False=负向点
    
    def __init__(self):
        super().__init__()
        self.image = None
        self.scaled_image = None
        self.prompt_manager = None
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setCursor(Qt.CursorShape.CrossCursor)
    
    def set_prompt_manager(self, manager):
        """关联提示管理器"""
        self.prompt_manager = manager
    
    def load_image(self, image_path: str):
        """加载图像"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            return False
        
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.fit_image_to_canvas()
        self.update()
        return True
    
    def fit_image_to_canvas(self):
        """缩放图像以适应画布"""
        if self.image is None:
            return
        
        canvas_w = self.width()
        canvas_h = self.height()
        img_h, img_w = self.image.shape[:2]
        
        # 计算缩放比例
        scale_w = canvas_w / img_w if img_w > 0 else 1
        scale_h = canvas_h / img_h if img_h > 0 else 1
        self.scale_factor = min(scale_w, scale_h, 1.0)
        
        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)
        
        self.scaled_image = cv2.resize(self.image, (new_w, new_h))
        
        # 计算偏移以居中显示
        self.offset.setX((canvas_w - new_w) // 2)
        self.offset.setY((canvas_h - new_h) // 2)
    
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放图像"""
        super().resizeEvent(event)
        if self.image is not None:
            self.fit_image_to_canvas()
            self.update()
    
    def paintEvent(self, event):
        """绘制画布内容"""
        super().paintEvent(event)
        painter = QPainter(self)
        
        # 绘制背景
        painter.fillRect(self.rect(), QColor('#1a1a1a'))
        
        # 绘制图像
        if self.scaled_image is not None:
            h, w = self.scaled_image.shape[:2]
            image_rgb = cv2.cvtColor(self.scaled_image, cv2.COLOR_RGB2BGR)
            bytes_per_line = 3 * w
            q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            painter.drawImage(self.offset, QPixmap.fromImage(q_image))
            
            # 绘制提示点
            if self.prompt_manager:
                self._draw_points(painter)
    
    def _draw_points(self, painter: QPainter):
        """绘制所有提示点"""
        if not self.prompt_manager or not self.scaled_image is not None:
            return
        
        for prompt in self.prompt_manager.prompts:
            color = QColor(prompt['color'])
            
            # 绘制正向点（圆形）
            for point in prompt['positive_points']:
                canvas_x = self.offset.x() + point[0] * self.scale_factor
                canvas_y = self.offset.y() + point[1] * self.scale_factor
                
                painter.setPen(QPen(QColor('white'), 2))
                painter.setBrush(color)
                painter.drawEllipse(QPoint(int(canvas_x), int(canvas_y)), 6, 6)
                
                # 序号
                painter.setPen(QPen(QColor('white'), 1))
                painter.setFont(QFont('Arial', 8))
                painter.drawText(int(canvas_x) + 8, int(canvas_y) - 8, '✓')
            
            # 绘制负向点（×符号）
            for point in prompt['negative_points']:
                canvas_x = self.offset.x() + point[0] * self.scale_factor
                canvas_y = self.offset.y() + point[1] * self.scale_factor
                
                painter.setPen(QPen(color, 3))
                r = 6
                painter.drawLine(
                    int(canvas_x - r), int(canvas_y - r),
                    int(canvas_x + r), int(canvas_y + r)
                )
                painter.drawLine(
                    int(canvas_x + r), int(canvas_y - r),
                    int(canvas_x - r), int(canvas_y + r)
                )
    
    def mousePressEvent(self, event):
        """鼠标点击事件"""
        if self.scaled_image is None or self.image is None:
            return
        
        # 转换到图像坐标
        img_x = (event.pos().x() - self.offset.x()) / self.scale_factor
        img_y = (event.pos().y() - self.offset.y()) / self.scale_factor
        
        # 检查是否在图像范围内
        h, w = self.scaled_image.shape[:2]
        if 0 <= img_x < (w / self.scale_factor) and 0 <= img_y < (h / self.scale_factor):
            img_x = int(img_x / self.scale_factor)
            img_y = int(img_y / self.scale_factor)
            
            if event.button() == Qt.MouseButton.LeftButton:
                # 正向点
                if self.prompt_manager:
                    self.prompt_manager.add_positive_point(img_x, img_y)
                self.point_added.emit(True)
            elif event.button() == Qt.MouseButton.RightButton:
                # 负向点
                if self.prompt_manager:
                    self.prompt_manager.add_negative_point(img_x, img_y)
                self.point_added.emit(False)
            
            self.update()
    
    def wheelEvent(self, event):
        """鼠标滚轮事件（预留用于缩放）"""
        pass


class PreviewCanvas(QFrame):
    """分割结果预览画布"""
    
    def __init__(self):
        super().__init__()
        self.preview_image = None
        self.offset = QPoint(0, 0)
        self.scale_factor = 1.0
        
        self.setStyleSheet("background-color: #1a1a1a;")
    
    def set_segmentation_result(self, result_image: np.ndarray):
        """设置分割结果图像"""
        self.preview_image = result_image.copy()
        self.fit_image_to_canvas()
        self.update()
    
    def fit_image_to_canvas(self):
        """缩放图像以适应画布"""
        if self.preview_image is None:
            return
        
        canvas_w = self.width()
        canvas_h = self.height()
        img_h, img_w = self.preview_image.shape[:2]
        
        scale_w = canvas_w / img_w if img_w > 0 else 1
        scale_h = canvas_h / img_h if img_h > 0 else 1
        self.scale_factor = min(scale_w, scale_h, 1.0)
        
        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)
        
        self.offset.setX((canvas_w - new_w) // 2)
        self.offset.setY((canvas_h - new_h) // 2)
    
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放图像"""
        super().resizeEvent(event)
        if self.preview_image is not None:
            self.fit_image_to_canvas()
            self.update()
    
    def paintEvent(self, event):
        """绘制预览"""
        super().paintEvent(event)
        painter = QPainter(self)
        
        painter.fillRect(self.rect(), QColor('#1a1a1a'))
        
        if self.preview_image is not None:
            h, w = self.preview_image.shape[:2]
            bytes_per_line = 3 * w
            q_image = QImage(self.preview_image.data, w, h, bytes_per_line, 
                            QImage.Format.Format_RGB888)
            painter.drawImage(self.offset, QPixmap.fromImage(q_image))


class PromptListWidget(QListWidget):
    """提示列表显示"""
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QListWidget {background-color: #2d2d2d; border: none;}
            QListWidget::item {padding: 8px; height: 32px;}
            QListWidget::item:selected {background-color: #1a2332;}
        """)


class SAM3SegmentUI(QMainWindow):
    """SAM3分割工具主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 分割工具")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet(self._get_stylesheet())
        
        # 初始化数据
        self.prompt_manager = PromptManager()
        self.current_image_path = None
        self.current_image = None
        self.segmentation_result = None
        
        # 初始化UI
        self._init_ui()
        self.prompt_manager.add_prompt()  # 初始化第一个提示组
        self._update_prompt_list()
    
    def _init_ui(self):
        """初始化UI布局"""
        # 中央Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 工具栏
        toolbar_layout = self._create_toolbar()
        main_layout.addLayout(toolbar_layout)
        
        # 主容器
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        # 左侧面板
        left_panel = self._create_left_panel()
        content_layout.addWidget(left_panel, 1)
        
        # 右侧面板
        right_panel = self._create_right_panel()
        content_layout.addWidget(right_panel, 1)
    
    def _create_toolbar(self) -> QVBoxLayout:
        """创建工具栏"""
        toolbar = QVBoxLayout()
        toolbar.setContentsMargins(12, 12, 12, 12)
        toolbar.setSpacing(8)
        
        # 第一行
        row1 = QHBoxLayout()
        
        self.upload_btn = self._create_button("📁 上传图像", self._on_upload_image)
        self.clear_prompt_btn = self._create_button("清除提示", self._on_clear_prompt, "danger")
        self.clear_all_btn = self._create_button("清除全部", self._on_clear_all, "danger")
        
        row1.addWidget(self.upload_btn)
        row1.addWidget(self.clear_prompt_btn)
        row1.addWidget(self.clear_all_btn)
        
        row1.addSpacing(20)
        
        self.status_label = QLabel("✓ 模型已加载")
        self.status_label.setStyleSheet("color: #888; font-size: 12px;")
        row1.addWidget(self.status_label)
        
        self.image_dims_label = QLabel("")
        self.image_dims_label.setStyleSheet("color: #888; font-size: 12px;")
        row1.addWidget(self.image_dims_label)
        
        row1.addStretch()
        
        self.segment_btn = self._create_button("🚀 开始切割", self._on_segment, "success")
        self.segment_btn.setEnabled(False)
        row1.addWidget(self.segment_btn)
        
        toolbar.addLayout(row1)
        
        return toolbar
    
    def _create_left_panel(self) -> QWidget:
        """创建左侧面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 图像画布
        self.image_canvas = ImageCanvas()
        self.image_canvas.set_prompt_manager(self.prompt_manager)
        self.image_canvas.point_added.connect(self._on_point_added)
        self.image_canvas.setMinimumHeight(300)
        layout.addWidget(self.image_canvas)
        
        # 提示管理面板
        prompt_panel = self._create_prompt_panel()
        layout.addWidget(prompt_panel)
        
        panel.setLayout(layout)
        return panel
    
    def _create_prompt_panel(self) -> QWidget:
        """创建提示管理面板"""
        panel = QFrame()
        panel.setStyleSheet("background-color: #2d2d2d; border-top: 1px solid #3d3d3d;")
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        # 标题
        title = QLabel("提示管理")
        title.setStyleSheet("font-weight: 600; font-size: 13px; color: #b0b0b0; text-transform: uppercase;")
        layout.addWidget(title)
        
        # 提示列表
        self.prompt_list = PromptListWidget()
        self.prompt_list.itemClicked.connect(self._on_prompt_selected)
        layout.addWidget(self.prompt_list)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        button_layout.setSpacing(6)
        
        clear_current_btn = self._create_button("清除当前提示", self._on_clear_prompt, "danger")
        clear_current_btn.setMaximumHeight(24)
        button_layout.addWidget(clear_current_btn)
        
        layout.addLayout(button_layout)
        
        # 添加提示组按钮
        add_prompt_btn = QPushButton("+ 添加Prompt组")
        add_prompt_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #07a77d;
                border: 2px dashed #07a77d;
                border-radius: 4px;
                padding: 8px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
        """)
        add_prompt_btn.setMaximumHeight(32)
        add_prompt_btn.clicked.connect(self._on_add_prompt)
        layout.addWidget(add_prompt_btn)
        
        panel.setLayout(layout)
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """创建右侧面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 预览画布
        self.preview_canvas = PreviewCanvas()
        self.preview_canvas.setMinimumHeight(300)
        layout.addWidget(self.preview_canvas)
        
        # 统计信息面板
        stats_panel = self._create_stats_panel()
        layout.addWidget(stats_panel)
        
        panel.setLayout(layout)
        return panel
    
    def _create_stats_panel(self) -> QWidget:
        """创建统计信息面板"""
        panel = QFrame()
        panel.setStyleSheet("background-color: #2d2d2d; border-top: 1px solid #3d3d3d;")
        layout = QGridLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # 统计项
        self.region_count_label = self._create_stat_item("📊 分割区域数", "0")
        self.processing_time_label = self._create_stat_item("⏱️ 处理时间", "-")
        self.active_prompt_label = self._create_stat_item("📌 活跃提示", "0/0")
        self.confidence_label = self._create_stat_item("【置信度】", "-")
        
        layout.addWidget(self.region_count_label, 0, 0)
        layout.addWidget(self.processing_time_label, 0, 1)
        layout.addWidget(self.active_prompt_label, 1, 0)
        layout.addWidget(self.confidence_label, 1, 1)
        
        panel.setLayout(layout)
        return panel
    
    def _create_stat_item(self, label: str, value: str) -> QFrame:
        """创建统计项"""
        item = QFrame()
        item.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border-left: 3px solid #0e639c;
                border-radius: 4px;
            }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        label_widget = QLabel(label)
        label_widget.setStyleSheet("color: #888; font-size: 11px; text-transform: uppercase;")
        
        value_widget = QLabel(value)
        value_widget.setStyleSheet("color: #fff; font-size: 13px; font-weight: 600;")
        value_widget.setObjectName("value")
        
        layout.addWidget(label_widget)
        layout.addWidget(value_widget)
        
        item.setLayout(layout)
        item.label = label_widget
        item.value = value_widget
        
        return item
    
    def _create_button(self, text: str, callback, style: str = "primary") -> QPushButton:
        """创建按钮"""
        btn = QPushButton(text)
        
        if style == "primary":
            bg = "#0e639c"
            hover = "#1177bb"
        elif style == "success":
            bg = "#07a77d"
            hover = "#0d9b6f"
        else:  # danger
            bg = "#d44e4e"
            hover = "#e85555"
        
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
            QPushButton:disabled {{
                background-color: #555;
                opacity: 0.6;
            }}
        """)
        btn.clicked.connect(callback)
        return btn
    
    def _get_stylesheet(self) -> str:
        """获取全局样式表"""
        return """
            QMainWindow {background-color: #1e1e1e; color: #e0e0e0;}
            QWidget {background-color: #1e1e1e; color: #e0e0e0;}
            QFrame {background-color: #1e1e1e;}
            QLabel {color: #e0e0e0;}
            QMessageBox {background-color: #2d2d2d;}
            QMessageBox QLabel {color: #e0e0e0;}
        """
    
    # ==================== 事件处理 ====================
    
    def _on_upload_image(self):
        """处理图像上传"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            if self.image_canvas.load_image(file_path):
                self.current_image_path = file_path
                self.current_image = cv2.imread(file_path)
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                
                h, w = self.current_image.shape[:2]
                self.image_dims_label.setText(f"{w} × {h}")
                self.segment_btn.setEnabled(True)
                
                self.prompt_manager.clear_all_prompts()
                self._update_prompt_list()
                
                self._show_message(f"✓ 图像已加载: {Path(file_path).name}", "info")
            else:
                self._show_message("✗ 图像加载失败", "error")
    
    def _on_add_prompt(self):
        """添加提示组"""
        if self.prompt_manager.add_prompt():
            self._update_prompt_list()
            self._show_message(f"✓ 已添加 Prompt {len(self.prompt_manager.prompts)}", "success")
        else:
            self._show_message("⚠️ 最多只能添加6个Prompt组", "warning")
    
    def _on_prompt_selected(self, item):
        """选择提示组"""
        index = self.prompt_list.row(item)
        self.prompt_manager.set_active_prompt(index)
        self._update_prompt_list()
        self.image_canvas.update()
    
    def _on_clear_prompt(self):
        """清除当前提示组"""
        count = self.prompt_manager.clear_current_prompt()
        self._update_prompt_list()
        self.image_canvas.update()
        self._show_message(f"✓ 已清除当前Prompt的 {count} 个点", "success")
    
    def _on_clear_all(self):
        """清除所有提示"""
        count = self.prompt_manager.clear_all_prompts()
        self._update_prompt_list()
        self.image_canvas.update()
        self._show_message(f"✓ 已清除所有 {count} 个点", "success")
    
    def _on_point_added(self, is_positive: bool):
        """点被添加"""
        self._update_prompt_list()
        self.image_canvas.update()
        point_type = "正向点" if is_positive else "负向点"
        self._show_message(f"✓ {point_type}已添加", "info")
    
    def _on_segment(self):
        """执行分割"""
        if self.current_image is None:
            self._show_message("⚠️ 请先上传图像", "warning")
            return
        
        if not self.prompt_manager.has_points():
            self._show_message("⚠️ 请标记至少一个提示点", "warning")
            return
        
        self.segment_btn.setEnabled(False)
        self._show_message("⏳ 处理中...", "info")
        
        # 记录开始时间
        start_time = datetime.now()
        
        # 模拟分割处理
        QApplication.processEvents()
        
        # 生成模拟分割结果
        self._generate_mock_segmentation()
        
        # 计算处理时间
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 更新预览
        self.preview_canvas.set_segmentation_result(self.segmentation_result['image'])
        
        # 更新统计信息
        self.processing_time_label.value.setText(f"{processing_time:.2f}s")
        self.region_count_label.value.setText(str(self.segmentation_result['regions']))
        self.confidence_label.value.setText(f"{self.segmentation_result['confidence']*100:.1f}%")
        
        self.segment_btn.setEnabled(True)
        self._show_message(f"✓ 分割完成 ({processing_time:.2f}s)", "success")
    
    def _generate_mock_segmentation(self):
        """生成模拟分割结果"""
        h, w = self.current_image.shape[:2]
        
        # 创建分割结果
        mask = np.zeros((h, w), dtype=np.uint8)
        result_image = self.current_image.copy()
        
        regions = min(5, len(self.prompt_manager.prompts) * 2)
        
        # 根据提示点生成分割区域
        for idx, prompt in enumerate(self.prompt_manager.prompts):
            if prompt['positive_points']:
                for point in prompt['positive_points']:
                    # 创建圆形区域
                    y, x = int(point[1]), int(point[0])
                    radius = 100
                    
                    for dy in range(-radius, radius):
                        for dx in range(-radius, radius):
                            py, px = y + dy, x + dx
                            if 0 <= px < w and 0 <= py < h:
                                if dx*dx + dy*dy < radius*radius:
                                    if mask[py, px] == 0:
                                        mask[py, px] = (idx % regions) + 1
        
        # 应用颜色到结果图像
        colors = [
            [255, 0, 255],    # Magenta
            [0, 255, 0],      # Green
            [255, 255, 0],    # Yellow
            [0, 255, 255],    # Cyan
            [255, 100, 0],    # Orange
        ]
        
        for i in range(1, regions + 1):
            region_mask = mask == i
            color = colors[(i - 1) % len(colors)]
            for c in range(3):
                result_image[region_mask, c] = int(result_image[region_mask, c] * 0.6 + color[c] * 0.4)
        
        self.segmentation_result = {
            'image': result_image,
            'mask': mask,
            'regions': regions,
            'confidence': 0.92
        }
    
    def _update_prompt_list(self):
        """更新提示列表显示"""
        self.prompt_list.clear()
        
        for idx, prompt in enumerate(self.prompt_manager.prompts):
            item = QListWidgetItem()
            pos_count = len(prompt['positive_points'])
            neg_count = len(prompt['negative_points'])
            
            text = f"Prompt {prompt['id']}  {pos_count}✓ {neg_count}✗"
            item.setText(text)
            
            if idx == self.prompt_manager.active_index:
                item.setSelected(True)
            
            self.prompt_list.addItem(item)
        
        # 更新活跃提示显示
        total = len(self.prompt_manager.prompts)
        active = self.prompt_manager.active_index + 1
        self.active_prompt_label.value.setText(f"{active}/{total}")
    
    def _show_message(self, text: str, msg_type: str = "info"):
        """显示消息提示"""
        if msg_type == "error":
            QMessageBox.critical(self, "错误", text)
        elif msg_type == "warning":
            QMessageBox.warning(self, "警告", text)
        else:
            QMessageBox.information(self, "提示", text)


def main():
    app = QApplication(sys.argv)
    window = SAM3SegmentUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
