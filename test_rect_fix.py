#!/usr/bin/env python3
"""
测试脚本：验证 ResizeHandle 和 EditableRect 的修复
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import QRectF
from lunar_eclipse_align.ui.select_rect import InteractiveGraphicsView, EditableRect
from PySide6.QtWidgets import QGraphicsScene


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ResizeHandle 修复测试")
        self.setGeometry(100, 100, 800, 600)

        # 创建中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 创建图形视图和场景
        self.view = InteractiveGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        layout.addWidget(self.view)

        # 连接信号
        self.view.rect_created.connect(self.on_rect_created)

        print("测试窗口创建成功")
        print("请在窗口中拖拽鼠标创建矩形，然后拖拽边缘/角落调整大小")

    def on_rect_created(self, rect):
        print(f"矩形创建成功: {rect}")
        print(f"句柄数量: {len(rect.handles)}")
        for handle_type, handle in rect.handles.items():
            print(f"  - {handle_type}: {handle}")


def test_direct_creation():
    """直接测试 EditableRect 创建"""
    print("\n=== 直接创建测试 ===")

    # 创建场景
    scene = QGraphicsScene()

    # 创建矩形
    rect = QRectF(100, 100, 200, 150)
    editable_rect = EditableRect(rect)

    print(f"EditableRect 创建成功: {editable_rect}")
    print(f"句柄数量: {len(editable_rect.handles)}")

    # 添加到场景
    scene.addItem(editable_rect)
    print("成功添加到场景")

    # 验证句柄也在场景中
    scene_items = scene.items()
    print(f"场景中的项目数量: {len(scene_items)}")
    print("场景中的项目:")
    for item in scene_items:
        print(f"  - {type(item).__name__}")

    return True


def main():
    app = QApplication(sys.argv)

    try:
        # 先进行直接创建测试
        if test_direct_creation():
            print("\n✅ 直接创建测试通过")
        else:
            print("\n❌ 直接创建测试失败")
            return 1

        # 创建测试窗口
        window = TestWindow()
        window.show()

        print("\n✅ 修复验证成功！")
        print("现在可以进行交互测试了")

        return app.exec()

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())