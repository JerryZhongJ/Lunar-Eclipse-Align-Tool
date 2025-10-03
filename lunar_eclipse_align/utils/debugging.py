import cv2
from numpy.typing import NDArray
from lunar_eclipse_align.utils.constants import DEBUG
from lunar_eclipse_align.utils.data_types import Circle


def debug_to_bgr(img_array: NDArray) -> NDArray:
    """将图像转换为RGB格式以便显示（仅在DEBUG模式下可用）"""
    if len(img_array.shape) == 2:  # 灰度图
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def debug_draw_circle(img_array: NDArray, circle: Circle) -> NDArray:
    """在图像上绘制圆（仅在DEBUG模式下可用）"""
    img_array = img_array.copy()
    center = (int(circle.x), int(circle.y))
    radius = int(circle.radius)
    cv2.circle(img_array, center, radius, (0, 0, 255), 2)  # 红色圆边
    cv2.circle(img_array, center, 2, (0, 0, 255), -1)  # 红色中心点
    return img_array


def debug_show(img_array: NDArray, circle: Circle | None):
    """使用OpenCV显示图像进行调试（仅在DEBUG模式下可用）"""
    img_array = debug_to_bgr(img_array)
    # 如果提供了圆，绘制圆
    if circle is not None:
        img_array = debug_draw_circle(img_array, circle)
    cv2.imshow("Debug View", img_array)
    cv2.waitKey(0)  # 等待任意按键
    cv2.destroyAllWindows()  # 关闭所有窗口


def debug_overlap(ref_img_array: NDArray, img_array: NDArray, circle: Circle | None):
    """使用OpenCV显示图像进行调试（仅在DEBUG模式下可用）"""
    if not DEBUG:
        return

    ref_img_array = debug_to_bgr(ref_img_array)
    img_array = debug_to_bgr(img_array)

    alpha = 0.5
    beta = 1.0 - alpha
    overlap_img = cv2.addWeighted(ref_img_array, alpha, img_array, beta, 0.0)

    # 如果提供了圆，绘制圆
    if circle is not None:
        overlap_img = debug_draw_circle(overlap_img, circle)
    cv2.imshow("Debug Overlap View", overlap_img)
    cv2.waitKey(0)  # 等待任意按键
    cv2.destroyAllWindows()  # 关闭所有窗口
