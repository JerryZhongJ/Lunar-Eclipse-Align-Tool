# ui_pyside6.py
"""
月食圆面对齐工具 - PySide6版本
UI界面模块
"""
import os
from pathlib import Path


from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSlider,
    QCheckBox,
    QTextBrowser,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QGroupBox,
)
from PySide6.QtCore import (
    Qt,
    Signal,
    QThread,
    QObject,
)


# 导入工具函数
from lunar_eclipse_align.pipeline import process_images
from lunar_eclipse_align.utils import (
    SUPPORTED_EXTS,
    SYSTEM,
    HoughParams,
)

from lunar_eclipse_align.ui_windows import PreviewWindow, DebugWindow, ProgressWindow


# 定义信号用于线程间通信
class ProgressSignal(QObject):
    """进度信号类"""

    progress_updated = Signal(int, str)  # 进度百分比，状态文本


class AlignmentThread(QThread):
    """对齐处理线程"""

    progress_signal = Signal(int, str)
    finished = Signal(bool, str)  # 是否成功，消息

    def __init__(
        self,
        in_path: Path,
        out_path: Path,
        hough: HoughParams,
        ref_path: Path | None,
        use_advanced,
        method,
        strong_denoise,
    ):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path
        self.hough = hough
        self.ref_path = ref_path
        self.use_advanced = use_advanced
        self.method = method
        self.strong_denoise = strong_denoise

    def run(self):
        """执行对齐处理"""
        try:
            # 创建进度回调函数
            def progress_callback(pct, status):
                self.progress_signal.emit(pct, status)

            # 执行对齐处理
            result = process_images(
                self.in_path,
                self.out_path,
                self.hough,
                # progress_callback,
                self.ref_path,
                self.use_advanced,
                self.strong_denoise,
            )

        except Exception as e:
            self.finished.emit(False, str(e))


class UniversalLunarAlignApp(QMainWindow):
    """月食圆面对齐工具主窗口"""

    def __init__(self):
        super().__init__()

        # 初始化变量
        self.preview_window = None
        self.progress_window = None
        self.debug_window = None
        self.alignment_thread = None
        self._about_photo = None
        self._qr_photo = None

        # 初始化UI变量
        self._init_variables()

        # 设置窗口
        self.setWindowTitle("月食圆面对齐工具")
        self.resize(920, 800)
        self.setMinimumSize(750, 700)

        # 创建UI
        self._setup_ui()

        # 设置初始日志信息
        self._set_initial_log_message()

        # 设置信号连接
        self._connect_signals()

    def _init_variables(self):
        """初始化变量"""
        self.input_path: Path | None = None
        self.output_path: Path | None = None
        self.reference_path: Path | None = None

        # 参数设置
        self.params = HoughParams(minRadius=300, maxRadius=800, param1=50, param2=30)

        self.use_advanced_alignment = False
        self.alignment_method = "auto"
        self.enable_strong_denoise = False

    def _setup_ui(self):
        """设置UI界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        # 创建各个UI区域
        self._create_path_section(main_layout)
        self._create_parameter_section(main_layout)
        self._create_debug_section(main_layout)
        self._create_action_section(main_layout)
        self._create_log_section(main_layout)

    def _create_path_section(self, parent_layout):
        """创建路径设置区域"""
        group = QGroupBox("1. 路径设置")
        layout = QGridLayout()
        group.setLayout(layout)

        # 输入文件夹
        layout.addWidget(QLabel("输入文件夹:"), 0, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("选择包含月食图像的文件夹...")
        layout.addWidget(self.input_edit, 0, 1)
        self.input_browse_btn = QPushButton("浏览...")
        self.input_browse_btn.clicked.connect(self.select_input_folder)
        layout.addWidget(self.input_browse_btn, 0, 2)

        # 输出文件夹
        layout.addWidget(QLabel("输出文件夹:"), 1, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("选择处理后图像的保存文件夹...")
        layout.addWidget(self.output_edit, 1, 1)
        self.output_browse_btn = QPushButton("浏览...")
        self.output_browse_btn.clicked.connect(self.select_output_folder)
        layout.addWidget(self.output_browse_btn, 1, 2)

        # 参考图像
        layout.addWidget(QLabel("参考图像:"), 2, 0)
        self.reference_edit = QLineEdit()
        self.reference_edit.setPlaceholderText("选择参考图像...")
        layout.addWidget(self.reference_edit, 2, 1)

        # 参考图像按钮布局
        ref_btn_layout = QHBoxLayout()
        ref_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.reference_select_btn = QPushButton("选择")
        self.reference_select_btn.clicked.connect(self.select_reference_image)
        ref_btn_layout.addWidget(self.reference_select_btn)
        self.reference_clear_btn = QPushButton("清除")
        self.reference_clear_btn.clicked.connect(self.clear_reference_image)
        ref_btn_layout.addWidget(self.reference_clear_btn)
        layout.addLayout(ref_btn_layout, 2, 2)

        # 帮助提示和强力降噪选项
        help_layout = QHBoxLayout()
        help_text = QLabel(
            "💡参考图像：作为对齐基准的图像。请在预览&半径估计窗口选择。"
        )
        help_text.setStyleSheet("color: gray; font-size: 10pt;")
        help_layout.addWidget(help_text)

        help_layout.addStretch()

        self.strong_denoise_check = QCheckBox("强力降噪(仅检测/对齐)")
        self.strong_denoise_check.setChecked(False)
        help_layout.addWidget(self.strong_denoise_check)

        layout.addLayout(help_layout, 3, 0, 1, 3)

        parent_layout.addWidget(group)

    def _create_parameter_section(self, parent_layout):
        """创建参数调节区域"""
        # 创建水平布局容器
        param_container = QWidget()
        param_layout = QHBoxLayout(param_container)
        param_layout.setContentsMargins(0, 0, 0, 0)

        # PHD2参数区域
        hough_group = QGroupBox("2. PHD2霍夫圆参数")
        hough_layout = QVBoxLayout(hough_group)

        # 帮助文本
        help_text = QLabel(
            "• PHD2增强算法：三级检测策略，自适应图像亮度\n"
            "• 最小/最大半径: 限制检测到的圆的半径范围(像素)\n"
            "• 参数1: Canny边缘检测高阈值\n"
            "• 参数2: 霍夫累加器阈值（关键参数）"
        )
        help_text.setStyleSheet("font-size: 9pt;")
        hough_layout.addWidget(help_text)

        # 参数控制
        param_widgets = {}
        param_configs = [
            ("minRadius", "最小半径:", 1, 3000),
            ("maxRadius", "最大半径:", 10, 4000),
            ("param1", "参数1 (Canny):", 1, 200),
            ("param2", "参数2 (累加阈值):", 1, 100),
        ]

        for i, (key, label, min_val, max_val) in enumerate(param_configs):
            # 创建参数行
            param_row = QWidget()
            param_row_layout = QHBoxLayout(param_row)
            param_row_layout.setContentsMargins(0, 0, 0, 0)

            # 标签
            param_label = QLabel(label)
            param_row_layout.addWidget(param_label, 1)

            # 滑块
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(self.params._asdict()[key])
            param_row_layout.addWidget(slider, 2)

            # 数值输入框
            spinbox = QSpinBox()
            spinbox.setMinimum(min_val)
            spinbox.setMaximum(max_val)
            spinbox.setValue(self.params._asdict()[key])
            param_row_layout.addWidget(spinbox, 0)

            # 连接信号
            slider.valueChanged.connect(
                lambda v, k=key, s=spinbox: self._on_param_changed(k, v, s)
            )
            spinbox.valueChanged.connect(
                lambda v, k=key, sl=slider: self._on_param_changed(k, v, sl)
            )

            # 保存控件引用
            param_widgets[key] = {"slider": slider, "spinbox": spinbox}

            hough_layout.addWidget(param_row)

        self.param_widgets = param_widgets
        param_layout.addWidget(hough_group, 2)

        # 多ROI精配准区域
        advanced_group = QGroupBox("3. 多ROI精配准")
        advanced_layout = QVBoxLayout(advanced_group)

        self.advanced_check = QCheckBox("启用多ROI精配准(仅支持赤道仪跟踪拍摄的素材)")
        self.advanced_check.setChecked(False)
        advanced_layout.addWidget(self.advanced_check)

        # 算法说明
        advanced_layout.addWidget(QLabel("算法说明:"))

        self.method_combo = QComboBox()
        self.method_combo.addItems(
            ["auto", "phase_corr", "template", "feature", "centroid"]
        )
        self.method_combo.setCurrentText("auto")
        self.method_combo.setEnabled(False)
        advanced_layout.addWidget(self.method_combo)

        # 算法帮助
        algo_help = QLabel(
            "• 在月盘内自动选择多块ROI进行 ZNCC/相位相关微调\n"
            "• 对亮度变化与阴影边界更鲁棒，失败时自动回退到圆心对齐\n"
            "• 建议在偏食/生光阶段启用，多数情况默认关闭即可"
        )
        algo_help.setStyleSheet("color: darkgreen; font-size: 8pt;")
        advanced_layout.addWidget(algo_help)

        warning = QLabel("⚠️ 实验性功能，不推荐开启")
        warning.setStyleSheet("color: orange; font-size: 9pt;")
        warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        advanced_layout.addWidget(warning)

        param_layout.addWidget(advanced_group, 1)

        parent_layout.addWidget(param_container)

    def _create_debug_section(self, parent_layout):
        """创建预览与调试区域"""
        group = QGroupBox("4. 预览与调试")
        layout = QHBoxLayout(group)

        self.preview_btn = QPushButton("打开预览 & 半径估计窗口")
        self.preview_btn.clicked.connect(self.open_preview)
        layout.addWidget(self.preview_btn, 1)

        self.debug_btn = QPushButton("打开调试窗口（实时参数预览）")
        self.debug_btn.clicked.connect(self.open_debug)
        layout.addWidget(self.debug_btn, 1)

        parent_layout.addWidget(group)

    def _create_action_section(self, parent_layout):
        """创建操作区域"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(200, 10, 200, 10)

        # 开始对齐按钮
        self.start_btn = QPushButton("🚀 开始集成对齐")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_alignment)
        layout.addWidget(self.start_btn, 1)

        # 打赏作者按钮
        self.donate_btn = QPushButton("打赏作者")
        self.donate_btn.clicked.connect(self.show_about_author)
        layout.addWidget(self.donate_btn, 0)

        parent_layout.addWidget(widget)

    def _create_log_section(self, parent_layout):
        """创建日志显示区域"""
        # 日志浏览器
        self.log_browser = QTextBrowser()
        self.log_browser.setReadOnly(True)
        self.log_browser.setMaximumHeight(200)

        parent_layout.addWidget(self.log_browser)

    def _connect_signals(self):
        """连接信号"""
        self.advanced_check.stateChanged.connect(self.on_advanced_changed)

    def _set_initial_log_message(self):
        """设置初始日志信息"""
        welcome = (
            f"欢迎使用月食圆面对齐工具 By @正七价的氟离子\n"
            f"运行平台: {SYSTEM}\n"
            "================================================================\n\n"
            "算法说明：\n"
            "• PHD2增强算法：基于霍夫圆检测，适用于完整清晰的月球\n"
            "• 多ROI精配准：适用于偏食、生光等复杂阶段（实验性）\n"
            "• 回退机制：确保在任何情况下都有可用的对齐方案\n\n"
            "使用建议：\n"
            "• 使用预览工具准确估算半径范围\n"
            "• 参数2（累加器阈值）是最关键的调整参数\n"
            f"• 支持格式：{', '.join(SUPPORTED_EXTS)}\n"
        )

        self.log_browser.append(welcome)

    def _on_param_changed(self, key, value, companion_widget):
        """参数改变时的处理"""
        setattr(self.params, key, value)

        # 阻止循环更新
        companion_widget.blockSignals(True)
        companion_widget.setValue(value)
        companion_widget.blockSignals(False)

    def on_advanced_changed(self):
        """高级功能状态改变"""
        enabled = self.advanced_check.isChecked()
        self.method_combo.setEnabled(enabled)

    def select_input_folder(self):
        """选择输入文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if folder:
            self.input_path = Path(folder)
            self.input_edit.setText(str(self.input_path))
            # 自动设置输出文件夹

            name = self.input_path.name
            output_dir = self.input_path.parent / f"{name}_aligned_v12b"
            self.output_path = output_dir
            self.output_edit.setText(str(self.output_path))

    def select_output_folder(self):
        """选择输出文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_path = Path(folder)
            self.output_edit.setText(self.output_path.name)

    def select_reference_image(self):
        """选择参考图像"""
        initial_dir = (
            self.input_path if self.input_path and self.input_path.is_dir() else Path()
        )
        file_filter = f"支持的图像 ( {' '.join(SUPPORTED_EXTS)} );;所有文件 (*.*)"

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择参考图像（用作对齐基准）", str(initial_dir), file_filter
        )

        file_path = Path(file_path)
        # 检查是否在输入文件夹内
        if self.input_path and not file_path.is_relative_to(self.input_path):
            reply = QMessageBox.question(
                self,
                "确认",
                "选择的参考图像不在输入文件夹内。\n建议选择输入文件夹中的图像作为参考。\n是否继续使用此图像？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.reference_path = file_path
        self.reference_edit.setText(file_path.name)

    def clear_reference_image(self):
        """清除参考图像"""
        self.reference_path = None
        self.reference_edit.setText("")

    def open_preview(self):
        """打开预览窗口"""
        if self.preview_window is None or not self.preview_window.isVisible():
            self.preview_window = PreviewWindow(self)
        self.preview_window.show()
        self.preview_window.raise_()
        self.preview_window.activateWindow()

    def open_debug(self):
        """打开调试窗口"""
        if self.debug_window is None or not self.debug_window.isVisible():
            self.debug_window = DebugWindow(self)
        self.debug_window.show()
        self.debug_window.raise_()
        self.debug_window.activateWindow()

    def _warning_dialog(self, title, message):
        """显示警告对话框"""
        return (
            QMessageBox.warning(
                self,
                title,
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        )

    def start_alignment(self):
        """开始对齐处理"""
        # 验证输入
        if not self.input_path or not os.path.isdir(self.input_path):
            QMessageBox.critical(self, "错误", "请选择有效的输入文件夹。")
            return

        if not self.output_path:
            QMessageBox.critical(self, "错误", "请指定输出文件夹。")
            return

        # 检查SciPy依赖
        use_advanced = self.advanced_check.isChecked()
        method = self.method_combo.currentText()

        # 检查参考图像
        ref_path = self.reference_path
        if ref_path and not os.path.exists(ref_path):
            if not self._warning_dialog(
                "警告",
                f"指定的参考图像不存在：\n{ref_path}\n\n是否继续（将自动选择参考图像）？",
            ):
                return
            else:
                ref_path = None

        # 准备霍夫参数

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.start_btn.setText(
            "集成对齐中 (多ROI + PHD2)..." if use_advanced else "PHD2对齐中..."
        )

        # 显示进度窗口
        pw = self.show_progress_window()

        # 创建并启动处理线程
        self.alignment_thread = AlignmentThread(
            self.input_path,
            self.output_path,
            self.params,
            ref_path,
            use_advanced,
            method,
            self.strong_denoise_check.isChecked(),
        )

        # 连接信号
        self.alignment_thread.progress_signal.connect(self.update_progress)
        self.alignment_thread.finished.connect(self.on_alignment_complete)

        # 启动线程
        self.alignment_thread.start()

    def update_progress(self, progress, status):
        """更新进度"""
        # 更新进度窗口
        if self.progress_window and self.progress_window.isVisible():
            self.progress_window.update_progress(progress, status)

        # 同时在日志中显示
        self.log_browser.append(f"进度: {progress}% - {status}")

    def show_progress_window(self):
        """显示进度窗口"""
        if self.progress_window is None or not self.progress_window.isVisible():
            self.progress_window = ProgressWindow(self)
        return self.progress_window

    def on_alignment_complete(self, success, message):
        """对齐完成"""
        # 恢复按钮状态
        self.start_btn.setEnabled(True)
        self.start_btn.setText("🚀 开始集成对齐")

        # 显示结果
        if success:
            QMessageBox.information(self, "处理完成", message)
        else:
            QMessageBox.critical(
                self, "处理失败", f"处理过程中发生错误，详情请查看日志。\n\n{message}"
            )

    def show_about_author(self):
        """显示关于作者窗口"""
        # TODO: 实现关于作者窗口
        QMessageBox.information(self, "关于作者", "关于作者功能正在开发中...")
