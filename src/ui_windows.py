# ui_windows_pyside6.py
"""
PySide6版本的窗口类
包含调试窗口、预览窗口、进度窗口等
"""
import os
from pathlib import Path
import threading
import queue
import numpy as np
from typing import TYPE_CHECKING

from image import ImageFile
from ui import UniversalLunarAlignApp

if TYPE_CHECKING:
    from ui import UniversalLunarAlignApp

from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSlider,
    QCheckBox,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QGraphicsView,
    QGraphicsScene,
    QApplication,
)
from PySide6.QtCore import (
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QFont,
    QPixmap,
    QImage,
    QColor,
    QPen,
    QBrush,
)


import cv2

# 导入工具函数
from utils import (
    SUPPORTED_EXTS,
    Hough,
    to_display_rgb,
)
import circle_detection as _algo_circle


class DebugWindow(QDialog):
    """调试窗口：可选择样张并实时调节参数"""

    def __init__(self, app_controller: UniversalLunarAlignApp):
        super().__init__(app_controller)
        self.app: UniversalLunarAlignApp = app_controller
        self.setWindowTitle("调试窗口（参数实时预览）")
        self.resize(980, 680)
        self.setMinimumSize(760, 520)

        # 图像数据
        self.preview_img_cv = None
        self.preview_gray_rgb = None
        self.preview_scale = 1.0
        self.current_path: Path | None = None

        # 调试计算控制
        self._dbg_queue = queue.Queue()
        self._dbg_worker = None
        self._dbg_cancel = threading.Event()
        self._dbg_job_id = 0
        self._dbg_busy = False
        self._dbg_pending = False
        self._last_det = None

        # 参数变量
        self.hough = Hough(
            minRadius=300,
            maxRadius=800,
            param1=50,
            param2=30,
        )

        self.show_mask: bool = False

        self._setup_ui()
        self._center_window()
        self._start_polling()

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

        self.choose_image_btn = QPushButton("选择样张")
        self.choose_image_btn.clicked.connect(self.choose_image)
        toolbar1_layout.addWidget(self.choose_image_btn)

        toolbar1_layout.addWidget(QLabel("最小半径:"))
        self.min_r_spin = QSpinBox()
        self.min_r_spin.setRange(1, 4000)
        self.min_r_spin.setValue(self.min_r)
        self.min_r_spin.valueChanged.connect(self.on_param_changed)
        toolbar1_layout.addWidget(self.min_r_spin)

        toolbar1_layout.addWidget(QLabel("最大半径:"))
        self.max_r_spin = QSpinBox()
        self.max_r_spin.setRange(1, 5000)
        self.max_r_spin.setValue(self.max_r)
        self.max_r_spin.valueChanged.connect(self.on_param_changed)
        toolbar1_layout.addWidget(self.max_r_spin)

        toolbar1_layout.addWidget(QLabel("参数1:"))
        self.param1_spin = QSpinBox()
        self.param1_spin.setRange(1, 200)
        self.param1_spin.setValue(self.param1)
        self.param1_spin.valueChanged.connect(self.on_param_changed)
        toolbar1_layout.addWidget(self.param1_spin)

        toolbar1_layout.addWidget(QLabel("参数2:"))
        self.param2_spin = QSpinBox()
        self.param2_spin.setRange(1, 100)
        self.param2_spin.setValue(self.param2)
        self.param2_spin.valueChanged.connect(self.on_param_changed)
        toolbar1_layout.addWidget(self.param2_spin)

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

    def on_param_changed(self, value):
        """参数改变"""
        self.min_r = self.min_r_spin.value()
        self.max_r = self.max_r_spin.value()
        self.param1 = self.param1_spin.value()
        self.param2 = self.param2_spin.value()
        self.refresh()

    def on_show_mask_changed(self, state):
        """显示分析区域状态改变"""
        self.show_mask = state == Qt.CheckState.Checked.value
        self.refresh()

    def choose_image(self):
        input_path = self.app.input_path
        """选择样张图像"""
        initial_dir = input_path if input_path and input_path.is_dir() else os.getcwd()
        file_filter = f"支持的图像 ( {' '.join(SUPPORTED_EXTS)} );;所有文件 (*.*)"

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择调试样张", str(initial_dir), file_filter
        )

        if file_path:
            self.current_path = Path(file_path)
            img = ImageFile(self.current_path).image
            if img is None:
                QMessageBox.critical(self, "错误", "无法读取该图像。")
                return

            # self.preview_img_cv = img

            self.preview_rgb = img.rgb

            # 生成灰度版
            try:
                if self.preview_rgb.ndim == 3:
                    gray = cv2.cvtColor(self.preview_rgb, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.preview_rgb
                self.preview_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            except Exception:
                self.preview_gray_rgb = self.preview_rgb

            # 设置回主界面
            self.refresh()

    def refresh(self):
        """刷新显示"""
        self.graphics_scene.clear()

        # 选择底图
        if self.show_mask:
            src_rgb = (
                self.preview_gray_rgb
                if self.preview_gray_rgb is not None
                else self.preview_rgb
            )
        else:
            src_rgb = self.preview_rgb

        if src_rgb is None:
            # 显示提示文字
            self.graphics_scene.addText("请选择一张样张…").setDefaultTextColor(
                "lightgray"
            )
            return

        # 复制底图
        display_rgb = src_rgb.copy()

        # 如果显示分析区域
        if self.show_mask and self.preview_img_cv is not None:
            try:
                gray = (
                    cv2.cvtColor(self.preview_img_cv, cv2.COLOR_BGR2GRAY)
                    if self.preview_img_cv.ndim == 3
                    else self.preview_img_cv
                )

                # 构建分析掩膜
                mask = self._build_analysis_mask(gray)
                if mask is not None:
                    H, W = display_rgb.shape[:2]
                    if mask.shape != (H, W):
                        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

                    # 创建红色叠加
                    red_overlay = display_rgb.copy()
                    red_overlay[:, :, 0] = 255
                    red_overlay[:, :, 1] = 0
                    red_overlay[:, :, 2] = 0

                    alpha = (mask.astype(np.float32) / 255.0) * 0.35
                    alpha = alpha[:, :, np.newaxis]

                    display_rgb = (
                        display_rgb * (1 - alpha) + red_overlay * alpha
                    ).astype(np.uint8)

            except Exception as e:
                print(f"显示分析区域失败: {e}")

        # 显示图像
        self._display_image(display_rgb)

        # 叠加检测结果
        if (
            self._last_det
            and isinstance(self._last_det, dict)
            and self._last_det.get("circle") is not None
        ):
            self._draw_detection_result(self._last_det)

    def _build_analysis_mask(self, gray):
        """构建分析掩膜"""
        try:
            # 尝试使用ui版本的掩膜构建
            try:
                return _algo_circle.build_analysis_mask_ui(
                    gray, brightness_min=3 / 255.0
                )
            except AttributeError:
                pass

            # 尝试新签名
            try:
                return _algo_circle.build_analysis_mask(gray, brightness_min=3 / 255.0)
            except TypeError:
                # 旧签名
                return _algo_circle.build_analysis_mask(gray)
        except Exception:
            # 回退：返回零掩膜
            return np.zeros_like(gray, dtype="uint8")

    def _display_image(self, rgb):
        """显示图像"""
        if rgb is None:
            return

        h, w = rgb.shape[:2]
        view_size = self.graphics_view.size()
        scale = min(view_size.width() / w, view_size.height() / h, 1.0)

        # 转换为QPixmap
        q_img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # 缩放
        if scale < 1.0:
            pixmap = pixmap.scaled(
                int(w * scale),
                int(h * scale),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # 添加到场景
        pixmap_item = self.graphics_scene.addPixmap(pixmap)
        self.preview_scale = scale

    def _draw_detection_result(self, det):
        """绘制检测结果"""
        try:
            cx, cy, r = det["circle"]

            # 处理缩放
            scale = self.preview_scale
            cx *= scale
            cy *= scale
            r *= scale

            # 画圆
            circle = self.graphics_scene.addEllipse(
                cx - r, cy - r, 2 * r, 2 * r, QPen(QColor(255, 77, 79), 2)
            )

            # 画圆心
            center = self.graphics_scene.addEllipse(
                cx - 3,
                cy - 3,
                6,
                6,
                QPen(Qt.PenStyle.NoPen),
                QBrush(QColor(255, 77, 79)),
            )

            # 显示状态文本
            quality = det.get("quality", None)
            status_text = f"检测到圆 r≈{r:.1f}px"
            if quality is not None:
                status_text += f"  quality={quality:.2f}"

            text = self.graphics_scene.addText(status_text)
            text.setFont(QFont("Arial", 12))
            text.setDefaultTextColor("lightgray")

        except Exception as e:
            print(f"绘制检测结果失败: {e}")

    def _start_debug_compute(self):
        """开始调试计算"""
        if self.preview_img_cv is None:
            return

        self._dbg_busy = True
        self._dbg_pending = False
        self._dbg_job_id += 1
        job_id = self._dbg_job_id

        img = self.preview_img_cv.copy()

        # 取消之前的任务
        if self._dbg_worker and self._dbg_worker.is_alive():
            self._dbg_cancel.set()

        self._dbg_cancel = threading.Event()
        self._last_det = None

        def _worker():
            try:
                # 执行检测
                det = self._detect_best(img)
                if det is not None:
                    self._dbg_queue.put((job_id, True, det))
                else:
                    self._dbg_queue.put((job_id, True, None))
            except Exception as e:
                print(f"调试计算失败: {e}")
                self._dbg_queue.put((job_id, False, None))

        self._dbg_worker = threading.Thread(target=_worker, daemon=True)
        self._dbg_worker.start()

    def _detect_best(self, img_bgr):
        """检测最佳圆"""
        # TODO: 实现圆检测逻辑
        return None

    def _start_polling(self):
        """开始轮询结果"""
        self._poll_debug_results()

    def _poll_debug_results(self):
        """轮询调试结果"""
        try:
            while True:
                job_id, success, det = self._dbg_queue.get_nowait()
                if job_id != self._dbg_job_id:
                    continue  # 过期任务

                self._dbg_busy = False
                if success:
                    self._last_det = det
                    self.refresh()
                break
        except queue.Empty:
            pass

        # 继续轮询
        QTimer.singleShot(40, self._poll_debug_results)


class PreviewWindow(QDialog):
    """预览窗口"""

    def __init__(self, app_controller: UniversalLunarAlignApp):
        super().__init__(app_controller)
        self.app: UniversalLunarAlignApp = app_controller
        self.setWindowTitle("预览与半径估计")
        self.resize(1100, 650)
        self.setMinimumSize(900, 500)

        self._center_window()
        self._setup_ui()

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

        # 工具条
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)

        self.choose_image_btn = QPushButton("选择样张")
        toolbar_layout.addWidget(self.choose_image_btn)

        toolbar_layout.addWidget(QLabel("增减范围 Δ:"))
        self.delta_spin = QSpinBox()
        self.delta_spin.setRange(0, 5000)
        self.delta_spin.setValue(100)
        toolbar_layout.addWidget(self.delta_spin)

        self.est_label = QLabel("估计半径: —")
        toolbar_layout.addWidget(self.est_label)

        toolbar_layout.addStretch()

        self.detect_btn = QPushButton("检测半径")
        self.detect_btn.clicked.connect(self.detect_radius)
        toolbar_layout.addWidget(self.detect_btn)

        layout.addWidget(toolbar)

        # 图像显示区域
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setBackgroundBrush(QBrush(QColor(51, 51, 51)))
        layout.addWidget(self.graphics_view)

        # 连接信号
        self.choose_image_btn.clicked.connect(self.choose_image)

    def choose_image(self):
        """选择预览图像"""
        input_path = self.app.input_path
        initial_dir = (
            self.app.input_path if input_path and input_path.is_dir() else Path()
        )
        file_filter = f"支持的图像 ( {' '.join(SUPPORTED_EXTS)} );;所有文件 (*.*)"

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择样张用于预览与框选", str(initial_dir), file_filter
        )

        if file_path:
            self.current_path = Path(file_path)
            img = ImageFile(self.current_path).image
            if img is None:
                QMessageBox.critical(self, "错误", "无法读取该图像。")
                return

            self.preview_img_rgb = img.rgb
            self.setWindowTitle(
                f"预览与半径估计 - {os.path.basename(self.current_path)}"
            )
            self._display_image()

    def _display_image(self):
        """显示图像"""
        if not hasattr(self, "preview_img_rgb") or self.preview_img_rgb is None:
            # 显示提示
            self.graphics_scene.clear()
            text = self.graphics_scene.addText("请选择样张，在图上拖拽鼠标框选月亮")
            text.setDefaultTextColor("lightgray")
            return

        # 转换为QPixmap
        h, w = self.preview_img_rgb.shape[:2]
        q_img = QImage(
            self.preview_img_rgb.data,
            w,
            h,
            self.preview_img_rgb.strides[0],
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(q_img)

        # 计算缩放
        view_size = self.graphics_view.size()
        scale = min(view_size.width() / w, view_size.height() / h, 1.0)
        if scale < 1.0:
            pixmap = pixmap.scaled(
                int(w * scale),
                int(h * scale),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # 添加到场景
        self.graphics_scene.clear()
        self.graphics_scene.addPixmap(pixmap)
        self.preview_scale = scale

    def detect_radius(self):
        """检测半径"""
        QMessageBox.information(self, "提示", "半径检测功能正在开发中...")


class ProgressWindow(QDialog):
    """进度窗口"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("处理进度")
        self.resize(400, 150)
        self.setFixedSize(400, 150)  # 固定大小

        self._setup_ui()
        self._center_window()

    def _center_window(self):
        """居中显示窗口"""
        screen = self.screen().geometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 2
        self.move(x, y)

    def _setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # 状态标签
        self.status_label = QLabel("准备开始...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        # 百分比标签
        self.percent_label = QLabel("0%")
        self.percent_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.percent_label)

    def update_progress(self, progress, status):
        """更新进度"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        self.percent_label.setText(f"{progress}%")
        QApplication.processEvents()  # 强制更新UI
