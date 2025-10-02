import logging
import os
from pathlib import Path
import threading
import queue
import cv2
import numpy as np
from typing import TYPE_CHECKING, Iterator

from lunar_eclipse_align.core.circle_detection import build_analysis_mask, detect_circle

from lunar_eclipse_align.utils.data_types import Circle
from lunar_eclipse_align.utils.image import Image, ImageFile


if TYPE_CHECKING:
    from lunar_eclipse_align.ui.main_window import UniversalLunarAlignApp
from numpy.typing import NDArray
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QGraphicsView,
    QGraphicsScene,
)
from PySide6.QtCore import (
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QPixmap,
    QImage,
    QColor,
    QPen,
    QBrush,
)

# 导入工具函数


class DebugWindow(QDialog):
    """调试窗口：可选择样张并实时调节参数"""

    def __init__(self, app_controller: "UniversalLunarAlignApp"):
        super().__init__(app_controller)
        self.app: "UniversalLunarAlignApp" = app_controller
        self.setWindowTitle("调试窗口（参数实时预览）")
        self.resize(980, 680)
        self.setMinimumSize(760, 520)

        # 图像数据
        self.preview_scale = 1.0

        self.current_dir: Path | None = None
        self.image_file_iterator: Iterator[tuple[Path, ImageFile]] | None = None
        self.current_path: Path | None = None
        self.current_img: Image | None = None
        self.image_files: dict[Path, ImageFile] = {}

        self.show_mask: bool = False

        self._setup_ui()
        self._center_window()

    def _center_window(self):
        """居中显示窗口"""
        self.adjustSize()
        screen = self.screen().geometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 2
        self.move(x, y)

    def _setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # 第一行工具条
        toolbar1 = QWidget()
        toolbar1_layout = QHBoxLayout(toolbar1)
        toolbar1_layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(toolbar1)

        # 第二行工具条
        toolbar2 = QWidget()
        toolbar2_layout = QHBoxLayout(toolbar2)
        toolbar2_layout.setContentsMargins(0, 0, 0, 0)

        self.show_mask_check = QCheckBox("显示分析区域")
        self.show_mask_check.setChecked(self.show_mask)
        self.show_mask_check.stateChanged.connect(self.on_show_mask_changed)
        toolbar2_layout.addWidget(self.show_mask_check)

        layout.addWidget(toolbar2)

        # 图像显示区域
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setBackgroundBrush(QBrush(QColor(34, 34, 34)))
        layout.addWidget(self.graphics_view)

    def showEvent(self, event):
        """窗口显示时自动弹出文件选择（仅当没有图像时）"""
        super().showEvent(event)
        if self.current_dir != self.app.input_path:
            self.refresh()
            self.current_dir = self.app.input_path

        if self.current_img is None:
            # pick one image in the input directory
            if not self.current_dir or not self.current_dir.is_dir():
                return
            self.image_files = ImageFile.load(self.current_dir)
            self.image_file_iterator = iter(self.image_files.items())
            try:
                self.current_path, img_file = next(self.image_file_iterator)
                self.current_img = img_file.image
                self.debug_display_image()
            except StopIteration:
                pass

    def on_show_mask_changed(self, state):
        """显示分析区域状态改变"""
        self.show_mask = state == Qt.CheckState.Checked.value
        self.refresh_pixmap()

    def debug_display_image(self):
        if self.current_img is None:
            return
        self.refresh_pixmap()
        circle = detect_circle(
            self.current_img,
            self.app.params,
            strong_denoise=self.app.enable_strong_denoise,
        )
        if circle:
            self._draw_circle(circle)

    def refresh(self):
        self.current_dir = None
        self.image_file_iterator = None
        self.current_img = None
        self.current_path = None
        self.image_files = {}

    def refresh_pixmap(self):
        """刷新显示"""
        self.graphics_scene.clear()
        if self.current_img is None:
            return
        display_rgb = self.current_img.rgb
        # 如果显示分析区域
        if self.show_mask:

            # 构建分析掩膜
            mask = build_analysis_mask(self.current_img, brightness_min=3 / 255.0)

            W, H = self.current_img.widthXheight
            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            # 创建红色叠加
            red_overlay = self.current_img.rgb.copy()
            red_overlay[:, :, 0] = 255
            red_overlay[:, :, 1] = 0
            red_overlay[:, :, 2] = 0

            alpha = (mask.astype(np.float32) / 255.0) * 0.35
            alpha = alpha[:, :, np.newaxis]

            display_rgb = (
                self.current_img.rgb * (1 - alpha) + red_overlay * alpha
            ).astype(np.uint8)
        # 显示图像
        self.display_pixmap(display_rgb)

    def display_pixmap(self, rgb: NDArray):
        """显示图像"""
        # 转换为QPixmap
        H, W, _ = rgb.shape[:2]
        q_img = QImage(
            rgb.data,
            W,
            H,
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(q_img)

        # 计算缩放
        view_size = self.graphics_view.size()
        scale = min(
            view_size.width() / W,
            view_size.height() / H,
            1.0,
        )
        if scale < 1.0:
            pixmap = pixmap.scaled(
                int(W * scale),
                int(H * scale),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # 清除场景并添加新图像
        self.graphics_scene.addPixmap(pixmap)
        self.preview_scale = scale

    def _draw_circle(self, circle: Circle):
        """绘制检测到的圆"""
        if self.drawed_circle:
            self.graphics_scene.removeItem(self.drawed_circle)
            self.drawed_circle = None
        if self.drawed_center:
            self.graphics_scene.removeItem(self.drawed_center)
            self.drawed_center = None

        # 应用预览缩放

        cx, cy, r = circle.x, circle.y, circle.radius
        # 画圆
        self.drawed_circle = self.graphics_scene.addEllipse(
            cx - r, cy - r, 2 * r, 2 * r, QPen(QColor(255, 77, 79), 2)
        )

        # 画圆心
        self.drawed_center = self.graphics_scene.addEllipse(
            cx - 3,
            cy - 3,
            6,
            6,
            QPen(Qt.PenStyle.NoPen),
            QBrush(QColor(255, 77, 79)),
        )
