#!/usr/bin/env python3
"""
月食圆面对齐工具 - PySide6版本
主程序入口
"""
import sys
import os

# 确保项目根目录在Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
except ImportError:
    print("错误：未安装PySide6。请运行: pip install PySide6")
    sys.exit(1)
from version import VERSION

def main():
    """主程序入口"""
    try:
        # 创建应用实例
        app = QApplication(sys.argv)

        # 设置应用信息
        app.setApplicationName("月食圆面对齐工具")
        app.setApplicationVersion(VERSION)
        app.setOrganizationName("正七价的氟离子")

        # 设置应用样式（自动使用系统原生样式）
        app.setStyle('Fusion')  # 跨平台现代样式

        # 启用高DPI支持
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)

        # 导入主窗口类（避免循环导入）
        from ui import UniversalLunarAlignApp

        # 创建主窗口
        window = UniversalLunarAlignApp()
        window.setWindowTitle(f"月食圆面对齐工具 V{VERSION} By @正七价的氟离子")
        window.resize(920, 800)
        window.setMinimumSize(750, 700)

        # 显示窗口
        window.show()

        # 运行应用
        sys.exit(app.exec())

    except Exception as e:
        print(f"程序启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()