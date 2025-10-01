from typing import Callable

from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsEllipseItem,
)
from PySide6.QtCore import (
    Qt,
    Signal,
    QRectF,
    QPointF,
)
from PySide6.QtGui import (
    QColor,
    QPen,
    QBrush,
)


class ResizeHandle(QGraphicsEllipseItem):
    """拖拽句柄类，用于调整矩形大小"""

    def __init__(self, parent_rect, handle_type, size=8):
        # 关键修改：将 parent_rect 作为 Qt 的父组件传入
        super().__init__(-size / 2, -size / 2, size, size, parent=parent_rect)
        self.parent_rect = parent_rect
        self.handle_type = handle_type  # 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'top', 'bottom', 'left', 'right'

        # 设置样式
        self.setPen(QPen(QColor(0, 191, 255), 2))  # 蓝色边框
        self.setBrush(QBrush(QColor(255, 255, 255)))  # 白色填充

        # 设置标志
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        # 设置鼠标样式
        self._set_cursor()

    def _set_cursor(self):
        """根据句柄类型设置鼠标样式"""
        cursor_map = {
            "top-left": Qt.CursorShape.SizeFDiagCursor,
            "top-right": Qt.CursorShape.SizeBDiagCursor,
            "bottom-left": Qt.CursorShape.SizeBDiagCursor,
            "bottom-right": Qt.CursorShape.SizeFDiagCursor,
            "top": Qt.CursorShape.SizeVerCursor,
            "bottom": Qt.CursorShape.SizeVerCursor,
            "left": Qt.CursorShape.SizeHorCursor,
            "right": Qt.CursorShape.SizeHorCursor,
        }
        self.setCursor(cursor_map.get(self.handle_type, Qt.CursorShape.ArrowCursor))

    def itemChange(self, change, value):
        """句柄位置改变时调整父矩形"""
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and self.parent_rect
        ):
            # 通知父矩形更新
            self.parent_rect.handle_moved(self.handle_type, value)
        return super().itemChange(change, value)


class EditableRect(QGraphicsRectItem):
    """可编辑的矩形框类"""

    def __init__(self, rect, parent=None):
        super().__init__(rect, parent)
        self.handles = {}
        self.is_updating = False

        # 设置样式
        self.setPen(QPen(QColor(0, 191, 255), 2))  # 蓝色边框
        self.setBrush(QBrush(QColor(0, 191, 255, 30)))  # 半透明蓝色填充

        # 设置标志
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        # 创建句柄
        self.create_handles()

        # 回调函数
        self.rect_changed_callback: Callable | None = None

    def create_handles(self):
        """创建8个拖拽句柄"""
        handle_types = [
            "top-left",
            "top",
            "top-right",
            "right",
            "bottom-right",
            "bottom",
            "bottom-left",
            "left",
        ]

        for handle_type in handle_types:
            handle = ResizeHandle(self, handle_type)
            self.handles[handle_type] = handle
            # 移除手动 addItem - Qt 父子关系会自动处理

        self.update_handles()

    def update_handles(self):
        """更新句柄位置"""
        if self.is_updating:
            return

        rect = self.rect()
        positions = {
            "top-left": QPointF(rect.left(), rect.top()),
            "top": QPointF(rect.center().x(), rect.top()),
            "top-right": QPointF(rect.right(), rect.top()),
            "right": QPointF(rect.right(), rect.center().y()),
            "bottom-right": QPointF(rect.right(), rect.bottom()),
            "bottom": QPointF(rect.center().x(), rect.bottom()),
            "bottom-left": QPointF(rect.left(), rect.bottom()),
            "left": QPointF(rect.left(), rect.center().y()),
        }

        for handle_type, handle in self.handles.items():
            if handle_type in positions:
                handle.setPos(positions[handle_type])

    def handle_moved(self, handle_type, new_pos):
        """处理句柄移动"""
        if self.is_updating:
            return

        self.is_updating = True
        rect = self.rect()

        # 根据句柄类型调整矩形
        if handle_type == "top-left":
            rect.setTopLeft(new_pos)
        elif handle_type == "top":
            rect.setTop(new_pos.y())
        elif handle_type == "top-right":
            rect.setTopRight(new_pos)
        elif handle_type == "right":
            rect.setRight(new_pos.x())
        elif handle_type == "bottom-right":
            rect.setBottomRight(new_pos)
        elif handle_type == "bottom":
            rect.setBottom(new_pos.y())
        elif handle_type == "bottom-left":
            rect.setBottomLeft(new_pos)
        elif handle_type == "left":
            rect.setLeft(new_pos.x())

        # 确保矩形有最小尺寸
        min_size = 20
        if rect.width() < min_size:
            if handle_type in ["top-left", "left", "bottom-left"]:
                rect.setLeft(rect.right() - min_size)
            else:
                rect.setRight(rect.left() + min_size)

        if rect.height() < min_size:
            if handle_type in ["top-left", "top", "top-right"]:
                rect.setTop(rect.bottom() - min_size)
            else:
                rect.setBottom(rect.top() + min_size)

        self.setRect(rect)
        self.update_handles()
        self.is_updating = False

        # 触发回调
        if self.rect_changed_callback:
            self.rect_changed_callback()

    def itemChange(self, change, value):
        """矩形位置改变时更新句柄"""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.update_handles()
            if self.rect_changed_callback:
                self.rect_changed_callback()
        return super().itemChange(change, value)

    def remove_from_scene(self):
        """从场景中移除矩形和所有句柄"""
        # 使用父子关系后，只需要移除父项，子项会自动被移除
        if self.scene():
            self.scene().removeItem(self)


class InteractiveGraphicsView(QGraphicsView):
    """交互式图形视图，支持鼠标绘制矩形框"""

    rect_created = Signal(object)  # 发射新创建的矩形信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.current_rect = None
        self.start_pos = None
        self.drawing = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # 转换到场景坐标
            scene_pos = self.mapToScene(event.pos())

            # 检查是否点击在现有矩形或句柄上
            item = self.scene().itemAt(scene_pos, self.transform())
            if isinstance(item, (EditableRect, ResizeHandle)):
                # 如果点击在现有元素上，交给默认处理
                super().mousePressEvent(event)
                return

            # 开始绘制新矩形
            self.start_pos = scene_pos
            self.drawing = True

            # 如果已有矩形，先移除
            if self.current_rect:
                self.current_rect.remove_from_scene()
                self.current_rect = None

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.start_pos:
            current_pos = self.mapToScene(event.pos())

            # 计算矩形区域
            rect = QRectF(self.start_pos, current_pos).normalized()

            # 更新或创建临时矩形
            if self.current_rect:
                self.current_rect.remove_from_scene()

            self.current_rect = EditableRect(rect)
            self.scene().addItem(self.current_rect)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False

            if self.current_rect:
                # 检查矩形大小是否合理
                rect = self.current_rect.rect()
                if rect.width() > 10 and rect.height() > 10:
                    # 发射信号，通知创建了新矩形
                    self.rect_created.emit(self.current_rect)
                else:
                    # 矩形太小，移除
                    self.current_rect.remove_from_scene()
                    self.current_rect = None

        super().mouseReleaseEvent(event)
