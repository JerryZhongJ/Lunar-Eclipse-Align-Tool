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

from lunar_eclipse_align.core.circle_detection import build_analysis_mask

from lunar_eclipse_align.core.image import ImageFile
from lunar_eclipse_align.ui.select_rect import EditableRect, InteractiveGraphicsView

if TYPE_CHECKING:
    from lunar_eclipse_align.ui.main_window import UniversalLunarAlignApp

from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QCheckBox,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsEllipseItem,
    QApplication,
)
from PySide6.QtCore import (
    Qt,
    QTimer,
    Signal,
    QRectF,
    QPointF,
)
from PySide6.QtGui import (
    QFont,
    QPixmap,
    QImage,
    QColor,
    QPen,
    QBrush,
    QCursor,
)


import cv2

# 导入工具函数
from lunar_eclipse_align.core.utils import (
    SUPPORTED_EXTS,
    HoughParams,
)


class DebugWindow(QDialog):
    """调试窗口：可选择样张并实时调节参数"""

    def __init__(self, app_controller: "UniversalLunarAlignApp"):
        super().__init__(app_controller)
        self.app: "UniversalLunarAlignApp" = app_controller
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
        self.hough = HoughParams(
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
                return build_analysis_mask(gray, brightness_min=3 / 255.0)
            except AttributeError:
                pass

            # 尝试新签名
            try:
                return build_analysis_mask(gray, brightness_min=3 / 255.0)
            except TypeError:
                # 旧签名
                return build_analysis_mask(gray)
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

    def __init__(self, app_controller: "UniversalLunarAlignApp"):
        super().__init__(app_controller)
        self.app: "UniversalLunarAlignApp" = app_controller
        self.setWindowTitle("预览与半径估计")
        self.resize(1100, 650)
        self.setMinimumSize(900, 500)

        # 图像相关
        self.current_path: Path | None = None
        self.preview_img_rgb = None
        self.preview_img_bgr = None  # 用于检测
        self.preview_scale = 1.0

        # 矩形选择相关
        self.current_rect: EditableRect | None = None
        self.detected_circle = None  # (cx, cy, r) 检测到的圆

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

        # 第一行工具条
        toolbar1 = QWidget()
        toolbar1_layout = QHBoxLayout(toolbar1)
        toolbar1_layout.setContentsMargins(0, 0, 0, 0)

        self.choose_image_btn = QPushButton("选择样张")
        self.choose_image_btn.clicked.connect(self.choose_image)
        toolbar1_layout.addWidget(self.choose_image_btn)

        toolbar1_layout.addWidget(QLabel("增减范围 Δ:"))
        self.delta_spin = QSpinBox()
        self.delta_spin.setRange(0, 5000)
        self.delta_spin.setValue(100)
        toolbar1_layout.addWidget(self.delta_spin)

        self.est_label = QLabel("估计半径: —")
        toolbar1_layout.addWidget(self.est_label)

        toolbar1_layout.addStretch()

        layout.addWidget(toolbar1)

        # 第二行工具条
        toolbar2 = QWidget()
        toolbar2_layout = QHBoxLayout(toolbar2)
        toolbar2_layout.setContentsMargins(0, 0, 0, 0)

        self.detect_btn = QPushButton("检测半径")
        self.detect_btn.clicked.connect(self.detect_radius)
        toolbar2_layout.addWidget(self.detect_btn)

        self.apply_btn = QPushButton("应用检测半径和参考图像")
        self.apply_btn.clicked.connect(self.apply_detected_radius)
        toolbar2_layout.addWidget(self.apply_btn)

        toolbar2_layout.addStretch()

        layout.addWidget(toolbar2)

        # 图像显示区域 - 使用新的交互式视图
        self.graphics_view = InteractiveGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setBackgroundBrush(QBrush(QColor(51, 51, 51)))

        # 连接矩形创建信号
        self.graphics_view.rect_created.connect(self.on_rect_created)

        layout.addWidget(self.graphics_view)

    def on_rect_created(self, rect: EditableRect):
        """当创建新矩形时的回调"""
        # 移除旧矩形
        if self.current_rect and self.current_rect != rect:
            self.current_rect.remove_from_scene()

        self.current_rect = rect

        # 设置矩形变化回调
        rect.rect_changed_callback = self.on_rect_changed

        # 立即计算估计半径
        self.on_rect_changed()

    def on_rect_changed(self):
        """矩形大小或位置改变时的回调"""
        if not self.current_rect:
            return

        rect = self.current_rect.rect()
        # 估计半径为矩形较小边的一半
        estimated_radius = min(rect.width(), rect.height()) / 2

        # 考虑预览缩放
        if self.preview_scale > 0:
            estimated_radius = estimated_radius / self.preview_scale

        self.est_label.setText(f"估计半径: {int(estimated_radius)} px")

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
            self.preview_img_bgr = img.bgr  # 保存BGR用于检测
            self.detected_circle = None  # 重置检测结果

            self.setWindowTitle(f"预览与半径估计 - {self.current_path.name}")
            self._display_image()

    def _display_image(self):
        """显示图像"""
        if self.preview_img_rgb is None:
            # 显示提示
            self.graphics_scene.clear()
            text = self.graphics_scene.addText("请选择样张，在图上拖拽鼠标框选月亮")
            text.setDefaultTextColor(QColor("lightgray"))
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

        # 清除场景并添加新图像
        self.graphics_scene.clear()
        self.graphics_scene.addPixmap(pixmap)
        self.preview_scale = scale

        # 重置当前矩形引用
        self.current_rect = None

        # 若有检测结果，重新绘制
        if self.detected_circle is not None:
            self._draw_detected_circle()

    def _draw_detected_circle(self):
        """绘制检测到的圆"""
        if self.detected_circle is None:
            return

        try:
            cx, cy, r = self.detected_circle

            # 应用预览缩放
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

        except Exception as e:
            print(f"绘制检测圆失败: {e}")

    def detect_radius(self):
        """检测半径"""
        if self.preview_img_bgr is None:
            QMessageBox.warning(self, "提示", "请先选择一张样张。")
            return

        if self.current_rect is None:
            QMessageBox.warning(self, "提示", "请先在图像上拖拽选择一个区域。")
            return

        # 获取矩形区域（场景坐标）
        rect = self.current_rect.rect()

        # 转换为图像坐标
        scale = self.preview_scale if self.preview_scale > 0 else 1.0
        img_rect = QRectF(
            rect.x() / scale,
            rect.y() / scale,
            rect.width() / scale,
            rect.height() / scale,
        )

        # 估计半径和中心
        estimated_radius = min(img_rect.width(), img_rect.height()) / 2
        center_x = img_rect.center().x()
        center_y = img_rect.center().y()

        # 实现圆检测算法（基于原始ui.py的logic）
        try:
            import cv2
            import numpy as np

            # 获取图像
            img = self.preview_img_bgr.copy()
            H0, W0 = img.shape[:2]

            # 缩放到合适大小以加快检测（类似原版逻辑）
            MAX_SIDE = 1600
            s = max(H0, W0) / float(MAX_SIDE)
            if s > 1.0:
                Hs = int(round(H0 / s))
                Ws = int(round(W0 / s))
                small = cv2.resize(img, (Ws, Hs), interpolation=cv2.INTER_AREA)
            else:
                s = 1.0
                small = img.copy()
                Hs, Ws = H0, W0

            # 转换为灰度
            if small.ndim == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            else:
                gray = small.copy()

            # 图像预处理
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # 计算检测参数（缩放到小图）
            min_r_s = max(1, int(round((estimated_radius * 0.8) / s)))
            max_r_s = max(min_r_s + 1, int(round((estimated_radius * 1.2) / s)))

            # 期望圆心（缩放到小图）
            exp_cx = center_x / s
            exp_cy = center_y / s

            # HoughCircles检测
            param1 = 50  # 可以从主界面获取
            param2 = 30
            minDist = max(30, min(gray.shape[:2]) // 4)

            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=min_r_s,
                maxRadius=max_r_s,
            )

            if circles is not None:
                circles = np.squeeze(circles, axis=0)

                # 选择最接近期望中心的圆
                best_circle = None
                min_dist = float("inf")

                for x, y, r in circles:
                    dist = np.sqrt((x - exp_cx) ** 2 + (y - exp_cy) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_circle = (x, y, r)

                if best_circle:
                    x, y, r = best_circle
                    # 还原到原图尺度
                    self.detected_circle = (x * s, y * s, r * s)
                else:
                    # 使用估计值
                    self.detected_circle = (center_x, center_y, estimated_radius)
            else:
                # 使用估计值
                self.detected_circle = (center_x, center_y, estimated_radius)

        except Exception as e:
            print(f"圆检测失败: {e}")
            # 回退到估计值
            self.detected_circle = (center_x, center_y, estimated_radius)

        # 重新绘制以显示检测结果
        self._display_image()

        # 更新标签
        _, _, r = self.detected_circle
        self.est_label.setText(f"估计半径: {int(r)} px (已检测)")

    def apply_detected_radius(self):
        """应用检测到的半径"""
        if self.detected_circle is None:
            QMessageBox.warning(self, "提示", "请先检测半径。")
            return

        _, _, r = self.detected_circle
        delta = self.delta_spin.value()

        min_r = max(1, int(r - delta))
        max_r = max(min_r + 1, int(r + delta))

        # 应用到主界面参数

        self.app.params.minRadius = min_r
        self.app.params.maxRadius = max_r

        # 设置参考图像
        if self.current_path:
            self.app.reference_path = self.current_path

        QMessageBox.information(
            self,
            "已应用",
            f"检测半径: {int(r)} px\n"
            f"设置范围: {min_r} - {max_r}\n"
            f"参考图像: {self.current_path.name if self.current_path else '无'}",
        )


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
