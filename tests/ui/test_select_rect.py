"""
UI组件单元测试 - select_rect模块
测试ResizeHandle和EditableRect的修复功能
"""

import pytest


@pytest.mark.unit
@pytest.mark.gui
def test_resize_handle_creation_with_parent():
    """测试ResizeHandle使用Qt父子关系创建（验证修复）"""
    from lunar_eclipse_align.ui.select_rect import EditableRect
    from PySide6.QtCore import QRectF
    from PySide6.QtWidgets import QGraphicsScene

    # 创建场景
    scene = QGraphicsScene()

    # 创建父矩形
    rect = QRectF(100, 100, 200, 150)
    editable_rect = EditableRect(rect)

    # 添加到场景
    scene.addItem(editable_rect)

    # 验证句柄已创建且有正确的父子关系
    assert len(editable_rect.handles) == 8

    # 验证每个句柄都有正确的父项
    for handle_type, handle in editable_rect.handles.items():
        assert handle.parentItem() == editable_rect
        assert handle.parent_rect == editable_rect
        assert handle.handle_type == handle_type

    # 验证句柄在场景中（通过父子关系自动添加）
    scene_items = scene.items()
    assert len(scene_items) == 9  # 1个矩形 + 8个句柄


@pytest.mark.unit
@pytest.mark.gui
def test_editable_rect_creation():
    """测试EditableRect创建和句柄自动生成"""
    from lunar_eclipse_align.ui.select_rect import EditableRect
    from PySide6.QtCore import QRectF

    # 创建矩形
    rect = QRectF(50, 50, 100, 80)
    editable_rect = EditableRect(rect)

    # 验证基本属性
    assert editable_rect.rect() == rect
    assert not editable_rect.is_updating
    assert editable_rect.rect_changed_callback is None

    # 验证句柄创建
    expected_handle_types = [
        "top-left",
        "top",
        "top-right",
        "right",
        "bottom-right",
        "bottom",
        "bottom-left",
        "left",
    ]
    assert len(editable_rect.handles) == 8
    assert set(editable_rect.handles.keys()) == set(expected_handle_types)

    # 验证每个句柄都正确配置
    for handle_type, handle in editable_rect.handles.items():
        assert handle.handle_type == handle_type
        assert handle.parent_rect == editable_rect


@pytest.mark.unit
def test_scene_additem_bug_fix():
    """回归测试：验证修复了'NoneType' has no attribute 'addItem'错误"""

    # 这个测试验证修复前的问题不再出现
    # 模拟修复前的情况：在没有场景的情况下创建EditableRect

    try:
        # 在没有PySide6的情况下，我们只能测试导入和基本结构
        from lunar_eclipse_align.ui.select_rect import EditableRect, ResizeHandle

        # 验证类可以正确导入
        assert EditableRect is not None
        assert ResizeHandle is not None

        # 验证ResizeHandle构造函数签名包含parent参数
        import inspect

        resize_handle_sig = inspect.signature(ResizeHandle.__init__)
        params = list(resize_handle_sig.parameters.keys())
        assert "parent_rect" in params

        # 验证EditableRect的create_handles方法不包含scene().addItem调用
        import ast
        import inspect

        # 获取create_handles方法的源码
        source = inspect.getsource(EditableRect.create_handles)

        # 解析AST查找是否有scene().addItem调用
        tree = ast.parse(source)

        has_scene_additem = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "addItem"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Attribute)
                and node.func.value.func.attr == "scene"
            ):
                has_scene_additem = True
                break

        # 验证已经移除了scene().addItem调用
        assert not has_scene_additem, "create_handles方法仍包含scene().addItem调用"

        print("✅ 回归测试通过：'NoneType' addItem错误已修复")

    except ImportError:
        # 如果PySide6不可用，跳过这个测试
        pytest.skip("PySide6 not available, skipping GUI regression test")


@pytest.mark.unit
def test_code_structure_validation():
    """验证代码结构修复"""
    from lunar_eclipse_align.ui.select_rect import ResizeHandle, EditableRect
    import inspect

    # 验证ResizeHandle构造函数包含parent参数
    resize_init_sig = inspect.signature(ResizeHandle.__init__)
    assert "parent_rect" in resize_init_sig.parameters

    # 验证EditableRect有正确的方法
    assert hasattr(EditableRect, "create_handles")
    assert hasattr(EditableRect, "update_handles")
    assert hasattr(EditableRect, "remove_from_scene")
    assert hasattr(EditableRect, "handle_moved")

    print("✅ 代码结构验证通过")
