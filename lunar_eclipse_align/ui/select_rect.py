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

    def __init__(self, parent_rect: "EditableRect", handle_type, size=8):
        # 关键修改：将 parent_rect 作为 Qt 的父组件传入
        super().__init__(-size / 2, -size / 2, size, size, parent=parent_rect)
        self.parent_rect = parent_rect
        self.handle_type = handle_type  # 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'top', 'bottom', 'left', 'right'
        self.dragging = False  # 添加拖拽状态标记

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
        """处理项变化事件"""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            new_pos = value
            self.parent_rect.handle_moved(self.handle_type, new_pos)
        return super().itemChange(change, value)


class EditableRect(QGraphicsRectItem):
    """可编辑的矩形框类"""

    def __init__(self, rect, parent=None):
        super().__init__(rect, parent)
        self.handles: dict[str, ResizeHandle] = {}
        self.is_updating = False

        # 设置样式
        self.setPen(QPen(QColor(0, 191, 255), 2))  # 蓝色边框
        self.setBrush(QBrush(QColor(0, 191, 255, 30)))  # 半透明蓝色填充

        # 设置标志
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

    def enable_handles(self):
        # 创建句柄
        self.create_handles()
        self.update_handles()

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

        self.update_handles()

    def update_handles(self):
        """更新句柄位置"""
        if self.is_updating:
            return  # 防止递归调用
        self.is_updating = True
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
        self.is_updating = False

    def handle_moved(self, handle_type, new_pos):
        """处理句柄移动"""

        rectF = self.rect()

        # 根据句柄类型调整矩形
        if handle_type == "top-left":
            rectF.setTopLeft(new_pos)
        elif handle_type == "top":
            rectF.setTop(new_pos.y())
        elif handle_type == "top-right":
            rectF.setTopRight(new_pos)
        elif handle_type == "right":
            rectF.setRight(new_pos.x())
        elif handle_type == "bottom-right":
            rectF.setBottomRight(new_pos)
        elif handle_type == "bottom":
            rectF.setBottom(new_pos.y())
        elif handle_type == "bottom-left":
            rectF.setBottomLeft(new_pos)
        elif handle_type == "left":
            rectF.setLeft(new_pos.x())

        self.setRect(rectF)
        self.update_handles()


class InteractiveGraphicsView(QGraphicsView):
    """交互式图形视图，支持鼠标绘制矩形框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.current_rect: EditableRect | None = None
        self.drawing_start: QPointF | None = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # 转换到场景坐标
            scene_pos = self.mapToScene(event.pos())

            if self.drawing_start:
                super().mousePressEvent(event)
                return  # 忽略多余的点击

            # 检查是否点击在现有矩形或句柄上
            item = self.scene().itemAt(scene_pos, self.transform())
            if isinstance(item, (EditableRect, ResizeHandle)):
                print("no")
                # 如果点击在现有元素上，交给默认处理
                super().mousePressEvent(event)
                return
            print("press")
            # 开始绘制新矩形
            self.drawing_start = scene_pos

            # 如果已有矩形，先移除
            if self.current_rect:
                self.scene().removeItem(self.current_rect)
            self.current_rect = EditableRect(QRectF(scene_pos, scene_pos))
            self.scene().addItem(self.current_rect)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing_start:
            assert self.current_rect is not None
            assert self.drawing_start is not None
            current_pos = self.mapToScene(event.pos())
            # 计算矩形区域
            rect = QRectF(self.drawing_start, current_pos).normalized()
            self.current_rect.setRect(rect)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing_start:
            print("release")
            assert self.current_rect is not None
            self.drawing_start = None

            # 检查矩形大小是否合理
            rect = self.current_rect.rect()
            if rect.width() > 10 and rect.height() > 10:
                self.current_rect.enable_handles()
            else:
                # 矩形太小，移除
                self.scene().removeItem(self.current_rect)
                self.current_rect = None

        super().mouseReleaseEvent(event)
