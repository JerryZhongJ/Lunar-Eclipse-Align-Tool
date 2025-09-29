from dataclasses import dataclass

from enum import Enum
import logging
import math, time
from typing import Any, Generator, Generic, TypeVar, overload
import numpy as np
import cv2
from numpy._typing._array_like import NDArray

from image import Image
from utils import Circle, DetectionResult, Hough, PositionArray, Vector, VectorArray
from skimage.measure import CircleModel, ransac


class BrightnessMode(Enum):
    AUTO = "auto"
    BRIGHT = "bright"
    NORMAL = "normal"
    DARK = "dark"


# ============== 预处理 & 质量评估 ==============


def adaptive_preprocessing(
    img: Image, brightness_mode: BrightnessMode
) -> tuple[NDArray, BrightnessMode]:
    """将图像转换为适合圆检测的灰度，并做适度增强。返回 (processed_gray, brightness_mode)"""

    gray = img.normalized_gray
    mean_brightness = float(np.mean(gray))
    if brightness_mode == BrightnessMode.AUTO:
        if mean_brightness > 140:
            brightness_mode = BrightnessMode.BRIGHT
        elif mean_brightness < 70:
            brightness_mode = BrightnessMode.DARK
        else:
            brightness_mode = BrightnessMode.NORMAL

    if brightness_mode == BrightnessMode.BRIGHT:
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    elif brightness_mode == BrightnessMode.DARK:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    return filtered, brightness_mode


def evaluate_circle_quality(img: Image, circle: Circle) -> float:
    """对检测到的圆做一个稳定的质量打分，越大越好（0~100）。"""

    w, h = img.widthXheight

    angles = np.linspace(0, 2 * np.pi, 48)
    directions = VectorArray(
        np.column_stack((np.cos(angles), np.sin(angles))), safe=False
    )
    inners: PositionArray = directions * (circle.radius - 2) + circle
    outers: PositionArray = directions * (circle.radius + 2) + circle
    mask = (
        0 <= inners.x < w
        and 0 <= inners.y < h
        and 0 <= outers.x < w
        and 0 <= outers.y < h
    )
    if not np.sum(mask):
        return 0.0

    inners = inners.filter(mask)
    outers = outers.filter(mask)

    gray = img.normalized_gray
    inner_vals = gray[inners.y.astype(int), inners.x.astype(int)]
    outer_vals = gray[outers.y.astype(int), outers.x.astype(int)]
    edge_strengths = np.abs(outer_vals - inner_vals)

    avg_edge = float(np.mean(edge_strengths))
    consistency = 1.0 / (1.0 + np.std(edge_strengths) / max(1.0, avg_edge))
    score = avg_edge * consistency
    return float(min(100.0, score))


# ============== 高光裁剪和星点抑制辅助 ==============
def clip_highlights(gray: NDArray, pct: float = 99.8):
    """Clip very bright highlights (glare/bloom) to a percentile to help Hough/RANSAC."""
    g = gray.astype(np.float32)
    cap = np.percentile(g, pct)
    if cap <= 0:
        return gray
    g = np.minimum(g, cap)
    g = g / (cap + 1e-6) * 255.0
    return g.astype(np.uint8)


def remove_stars_small(gray: NDArray):
    """Suppress point-like stars/noise while preserving lunar rim."""
    # Top-hat to remove small bright dots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    g = cv2.subtract(gray, tophat)
    # Gentle median to clean salt-pepper
    g = cv2.medianBlur(g, 3)
    return g


# ============== 稳健外缘 RANSAC（用于血月/缺口） ==============


def edge_points_outer_rim(
    gray: np.ndarray, prev_circle: Circle | None = None
) -> PositionArray | None:
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return None

    pts = PositionArray(np.stack([xs, ys], axis=1), safe=False)
    if prev_circle is None:
        return pts
    vectors = pts - prev_circle
    distance_cond = (
        prev_circle.radius * 0.85 < vectors.norms() < prev_circle.radius * 1.15
    )

    valid_pts = pts.filter(distance_cond)
    valid_vectors = vectors.filter(distance_cond)
    valid_x = valid_vectors.x.astype(np.int32)
    valid_y = valid_vectors.y.astype(np.int32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    valid_gx = gx[valid_y, valid_x]
    valid_gy = gy[valid_y, valid_x]
    gradients = VectorArray(np.stack([valid_gx, valid_gy], axis=1), safe=False)

    cond = gradients.normalize() * valid_vectors.normalize() > 0.2
    keep = valid_pts.filter(cond)
    if len(keep) < 30:
        return None
    return keep


# ============== 遮罩相位相关（亚像素平移微调） ==============


def masked_phase_corr(img: Image, ref_img: Image, circle: Circle) -> Vector:
    W, H = ref_img.widthXheight
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X - circle.x) ** 2 + (Y - circle.y) ** 2)

    mask = (dist <= circle.radius * 0.98).astype(np.float32)
    band = (dist >= circle.radius * 0.90) & (dist <= circle.radius * 0.98)
    t = (dist[band] - circle.radius * 0.90) / (circle.radius * 0.08 + 1e-6)
    mask[band] = 0.5 * (1 + np.cos(np.pi * (1 - t)))

    rg = (ref_img.normalized_gray * mask).astype(np.float32)
    tg = (img.normalized_gray * mask).astype(np.float32)

    (dx, dy), _ = cv2.phaseCorrelate(rg, tg)
    return Vector(dx, dy)


# ============== 辅助：粗估 & 环形 ROI（抑制星点/加速霍夫） ==============


def rough_center_radius(gray: NDArray, min_r: float, max_r: float) -> Circle | None:
    g = cv2.GaussianBlur(gray, (0, 0), 2.0)
    # Use adaptive + Otsu fallback to handle glare/crescent
    thr = max(10, int(np.mean(g) + 0.3 * np.std(g)))
    _, bw1 = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
    _, bw2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_adap = cv2.adaptiveThreshold(
        g.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -5
    )
    bw = cv2.max(bw1, cv2.max(bw2, bw_adap))
    bw = cv2.morphologyEx(
        src=bw,
        op=cv2.MORPH_OPEN,
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    if min_r * 0.6 <= r <= max_r * 1.6:
        return Circle(cx, cy, r)
    return None


def ring_mask(
    width: int, height: int, circle: Circle, inner=0.70, outer=1.15
) -> NDArray[np.uint8]:
    Y, X = np.ogrid[:height, :width]
    dist = np.sqrt((X - circle.x) ** 2 + (Y - circle.y) ** 2)
    m = ((dist >= circle.radius * inner) & (dist <= circle.radius * outer)).astype(
        np.uint8
    ) * 255
    return m


# ============== UI 调参可视化：分析区域掩膜（仅供显示） ==============
def build_analysis_mask(
    img_gray: NDArray,
    brightness_min=3 / 255.0,
    min_radius: float | None = None,
    max_radius: float | None = None,
):
    """
    仅供 UI 调参窗口显示“分析区域”用：
    - uint8 归一化 -> 轻度去噪
    - Otsu 阈值 与 亮度下限并联
    - 形态学开运算清点
    - 仅保留最大连通域
    返回 bool(H,W)。不影响主流程检测。
    """
    try:
        g = img_gray.copy()
        if g.dtype != np.uint8:
            cv2.normalize(
                g.astype(np.float32), g, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
        g = cv2.GaussianBlur(g, (3, 3), 0)
        _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        floor_t = max(1, int(round(float(brightness_min) * 255.0)))
        _, floor = cv2.threshold(g, floor_t, 255, cv2.THRESH_BINARY)
        m = cv2.bitwise_and(otsu, floor)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return np.zeros_like(m, dtype=bool)
        c = max(cnts, key=cv2.contourArea)
        keep = np.zeros_like(m)
        cv2.drawContours(keep, [c], -1, 255, thickness=cv2.FILLED)
        return keep.astype(bool)
    except Exception:
        # 兜底：整图 False
        shape = (1, 1) if img_gray is None else img_gray.shape[:2]
        return np.zeros(shape, dtype=bool)


# 兼容 UI 中的优先调用名
build_analysis_mask_ui = build_analysis_mask

# ============== 主检测（供 pipeline 调用） ==============


def touches_border(img: Image, circle: Circle, margin: int = 5) -> bool:
    if circle is None:
        return True
    w, h = img.widthXheight
    return (
        (circle.x - circle.radius < margin)
        or (circle.y - circle.radius < margin)
        or (circle.x + circle.radius > w - margin)
        or (circle.y + circle.radius > h - margin)
    )


# ------------------ 检测环境设置 ------------------
def _setup_detection_environment(
    img: Image, strong_denoise: bool = False
) -> tuple[NDArray, BrightnessMode]:
    """
    设置检测环境，包括预处理和降噪

    Args:
        image: 输入图像
        strong_denoise: 是否使用强力降噪

    Returns:
        tuple: (processed, processed_det, proc_for_hough, brightness_mode, H, W)
    """

    processed, brightness_mode = adaptive_preprocessing(img, BrightnessMode.AUTO)

    # 可选：强力降噪（仅影响检测，不影响最终成片）
    if strong_denoise:

        # fast NLM 能在强噪场景下保持边缘
        processed = cv2.fastNlMeansDenoising(
            processed, None, h=10, templateWindowSize=7, searchWindowSize=21
        )
        # 轻度中值进一步压盐胡椒
        processed = cv2.medianBlur(processed, 3)

    processed = clip_highlights(processed, pct=99.8)
    # Use a detection-optimized copy to make Hough/RANSAC more stable on glare/bloom frames
    processed = remove_stars_small(processed)

    return processed, brightness_mode


def hough_on_thumb_detect(img: Image, processed: NDArray, hough: Hough) -> list[Circle]:
    max_side = max(img.width, img.height)
    circles = []
    if max_side <= 1800:
        return []
    scale = 1800.0 / max_side
    small = cv2.resize(
        processed,
        (int(img.width * scale), int(img.height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    new_hough = Hough(
        minRadius=max(1, int(hough.minRadius * scale)),
        maxRadius=max(2, int(hough.maxRadius * scale)),
        param1=hough.param1,
        param2=max(hough.param2 - 5, 10),
    )

    sc = cv2.HoughCircles(
        small,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=small.shape[0] // 2,
        **new_hough._asdict(),
    )
    if sc is None:
        return []
    circles = [Circle.from_ndarray(c / scale) for c in sc[0]]
    logging.info("缩放霍夫(thumb)")
    return circles


# ------------------ ROI构建 ------------------
def build_detection_roi(
    img: Image,
    processed: NDArray,
    hough: Hough,
    prev_circle: Circle | None,
) -> NDArray:

    # —— 粗估中心半径，构建环形 ROI —— #
    est: Circle | None = rough_center_radius(
        processed, hough.minRadius, hough.maxRadius
    )
    if est:
        ring = ring_mask(img.width, img.height, est, inner=0.70, outer=1.15)
        processed = cv2.bitwise_and(processed, processed, mask=ring)

    # —— 若给出上一帧圆心半径，合并一个"历史先验"环形 ROI —— #
    if prev_circle:

        inner = max(
            0.70,
            min(0.85, (hough.minRadius / max(prev_circle.radius, 1e-6)) * 0.9),
        )
        outer = min(
            1.30,
            max(1.15, (hough.maxRadius / max(prev_circle.radius, 1e-6)) * 1.05),
        )
        ring_prev = ring_mask(
            img.width, img.height, prev_circle, inner=inner, outer=outer
        )
        processed = cv2.bitwise_and(processed, processed, mask=ring_prev)
        return processed

    return processed


# ------------------ 超时检测函数 ------------------
def timeout_fallback_detection(
    img: Image, processed: NDArray, hough: Hough
) -> list[Circle]:
    """
    超时时的降级检测

    Args:
        processed_det: 检测优化图像
        processed: 处理后的图像
        hough: 霍夫参数
        H, W: 图像尺寸

    Returns:
        tuple: (best_circle, best_score)
    """

    scale = min(1.0, 1600.0 / max(img.height, img.width))
    small = cv2.resize(
        processed,
        (int(img.width * scale), int(img.height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    new_hough = Hough(
        minRadius=max(1, int(hough.minRadius * scale)),
        maxRadius=max(2, int(hough.maxRadius * scale)),
        param1=max(hough.param1, 20),
        param2=max(hough.param2 - 5, 8),
    )
    sc = cv2.HoughCircles(
        small,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=small.shape[0] // 2,
        **new_hough._asdict(),
    )
    if sc is None:
        return []
    circles = [Circle.from_ndarray(c) for c in sc[0]]

    logging.info("超时降级(thumb)")

    return circles


# ------------------ 稳健RANSAC检测 ------------------
def robust_ransac_detect(
    processed: NDArray,
) -> list[Circle]:
    """
    尝试稳健外缘RANSAC检测

    Args:
        processed_det: 检测优化图像
        processed: 处理后的图像
        best_score: 当前最佳分数

    Returns:
        tuple: (best_circle, best_score)
    Raises:
        Exception: 如果检测失败
    """

    robust = detect_circle_robust(processed, None)
    if robust is None:
        return []
    logging.info("稳健外缘RANSAC")

    return [robust]


def detect_circle_robust(
    gray: np.ndarray, prev_circle: Circle | None = None
) -> Circle | None:
    pts = edge_points_outer_rim(gray, prev_circle)
    if not pts:
        return prev_circle
    model, inliers = ransac(
        data=pts._arr,
        model_class=CircleModel,
        min_samples=3,
        residual_threshold=2.0,
        max_trials=120,
        stop_probability=0.99,  # type: ignore
    )
    if np.sum(inliers) < 40:  # type: ignore
        return prev_circle
    cy, cx, r = model.params  # type: ignore
    cand = Circle(float(cx), float(cy), float(r))

    vectors = pts - cand
    arctans = np.arctan2(vectors.y, vectors.x)
    span: np.float64 = np.ptp(arctans)
    if prev_circle and span < (2 * np.pi / 3.0):  # <120°
        cand = Circle(cand.x, cand.y, prev_circle.radius)
    return cand


# ------------------ 标准霍夫检测 ------------------
def standard_hough_detect(
    img: Image, proc_for_hough: NDArray, hough: Hough
) -> list[Circle]:
    """
    尝试标准霍夫检测

    Args:
        proc_for_hough: 用于霍夫变换的图像
        processed: 处理后的图像
        hough: 霍夫参数
        best_score: 当前最佳分数
        height: 图像高度

    Returns:
        tuple: (best_circle, best_score)
    Raises:
        Exception: 如果检测失败
    """

    circles = cv2.HoughCircles(
        proc_for_hough,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=img.height,
        **hough._asdict(),
    )
    if circles is None:
        return []
    logging.info("标准霍夫")
    circles = [Circle(*c) for c in circles[0]]

    return circles


# ------------------ 自适应霍夫检测 ------------------
def adaptive_hough_detect(
    img: Image,
    proc_for_hough: NDArray,
    hough: Hough,
    brightness_mode: BrightnessMode,
) -> list[Circle]:
    """
    尝试自适应霍夫检测，根据亮度模式调整参数

    Args:
        proc_for_hough: 用于霍夫变换的图像
        processed: 处理后的图像
        hough: 霍夫参数
        brightness_mode: 亮度模式
        best_score: 当前最佳分数
        height: 图像高度

    Returns:
        tuple: (best_circle, best_score)
    """

    if brightness_mode == BrightnessMode.BRIGHT:
        new_hough = Hough(
            minRadius=hough.minRadius,
            maxRadius=hough.maxRadius,
            param1=hough.param1 + 20,
            param2=max(hough.param2 - 5, 10),
        )
    elif brightness_mode == BrightnessMode.DARK:
        new_hough = Hough(
            minRadius=hough.minRadius,
            maxRadius=hough.maxRadius,
            param1=max(hough.param1 - 15, 20),
            param2=max(hough.param2 - 10, 5),
        )
    else:
        new_hough = Hough(
            minRadius=hough.minRadius,
            maxRadius=hough.maxRadius,
            param1=hough.param1,
            param2=max(hough.param2 - 8, 8),
        )

    circles = cv2.HoughCircles(
        proc_for_hough,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=img.height // 2,
        **new_hough._asdict(),
    )
    if circles is None:
        return []
    circles = [Circle(*c) for c in circles[0]]

    logging.info(f"自适应霍夫(P1={new_hough.param1},P2={new_hough.param2})")

    return circles


# ------------------ 轮廓检测 ------------------
def contour_detect(processed_det: NDArray, hough: Hough) -> list[Circle]:
    """
    尝试轮廓检测作为备选方案

    Args:
        processed_det: 检测优化图像
        processed: 处理后的图像
        hough: 霍夫参数
        best_score: 当前最佳分数

    Returns:
        tuple: (best_circle, best_score)
    """

    mean_val = float(np.mean(processed_det))
    tv = max(50, int(mean_val * 0.7))
    _, binary = cv2.threshold(processed_det, tv, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = hough.minRadius**2 * np.pi * 0.3
    max_area = hough.maxRadius**2 * np.pi * 2.0
    contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
    circles = [
        Circle(c[0][0], c[0][1], c[1])
        for cnt in contours
        if (c := cv2.minEnclosingCircle(cnt))
    ]
    circles = [c for c in circles if hough.minRadius <= c.radius <= hough.maxRadius]
    logging.info(f"轮廓检测(T={tv})")

    return circles


# ------------------ Padding 降级检测 ------------------
def padding_fallback_detect(
    processed: NDArray,
    hough: Hough,
) -> list[Circle]:
    """
    尝试基于 padding 的降级检测，用于处理边界情况

    Args:
        processed_det: 检测优化图像
        hough: 霍夫参数
        H, W: 图像尺寸

    Returns:
        tuple: (best_circle, best_score)
    """

    pad = int(max(32, round(hough.maxRadius * 1.2)))
    processed_pad = cv2.copyMakeBorder(
        processed,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    # Use constant black padding to avoid mirrored ghosts influencing Hough
    est_p = rough_center_radius(
        processed_pad, int(hough.minRadius * 1.1), int(hough.maxRadius * 1.1)
    )
    if est_p is not None:
        ring_p = ring_mask(
            processed_pad.shape[0],
            processed_pad.shape[1],
            est_p,
            inner=0.70,
            outer=1.15,
        )
        proc_pad_for_hough = cv2.bitwise_and(processed_pad, processed_pad, mask=ring_p)
    else:
        proc_pad_for_hough = processed_pad
    new_hough = Hough(
        minRadius=int(max(1, hough.minRadius * 1.1)),
        maxRadius=int(max(2, hough.maxRadius * 1.1)),
        param1=max(hough.param1, 20),
        param2=max(hough.param2 - 5, 8),
    )
    circles_p = cv2.HoughCircles(
        proc_pad_for_hough,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=processed_pad.shape[0] // 2,
        **new_hough._asdict(),
    )

    if not circles_p:
        robust_p = detect_circle_robust(processed_pad)
        if robust_p:
            circles = [robust_p]
        else:
            circles = []
    else:
        circles = [Circle.from_ndarray(c) for c in circles_p[0]]

    return circles


# ------------------ 最终圆验证 ------------------
def final_detect(
    processed_det: NDArray,
    hough: Hough,
) -> list[Circle]:
    """
    最终半径窗口一致性检查，严格遵守UI设定

    Args:
        best_circle: 当前最佳圆
        best_score: 当前最佳分数
        processed_det: 检测优化图像
        processed: 处理后的图像
        hough: 霍夫参数

    Returns:
        tuple: (best_circle, best_score)
    """

    # 在严格窗口内再做一次快速霍夫重试
    height, width = processed_det.shape
    _minDist_coreS = max(16, min(height, width) // 4)
    new_hough = Hough(
        minRadius=max(1, hough.minRadius),
        maxRadius=hough.maxRadius,
        param1=max(hough.param1, 20),
        param2=max(hough.param2 - 5, 8),
    )
    scS = cv2.HoughCircles(
        processed_det,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=_minDist_coreS,
        **new_hough._asdict(),
    )
    if scS is None:
        return []
    circles = [Circle.from_ndarray(c) for c in scS[0]]
    circles = [c for c in circles if hough.minRadius <= c.radius <= hough.maxRadius]

    return circles


def detect_circle(
    img: Image,
    hough: Hough,
    strong_denoise=False,
    prev_circle: Circle | None = None,
) -> DetectionResult | None:
    """
    主检测函数 - 重构后的协调器版本

    将原本的单体大函数拆分为多个职责明确的辅助函数
    """

    best_circle: Circle | None = None
    best_score: float = 0.0

    # 1. 设置检测环境
    processed, brightness_mode = _setup_detection_environment(img, strong_denoise)

    def update_best(circles: list[Circle]) -> None:
        nonlocal best_circle, best_score, processed
        for c in circles:
            score = evaluate_circle_quality(img, c)
            if score > best_score:
                best_circle = c
                best_score = score

    # 2. 构建检测ROI
    proc_for_hough = build_detection_roi(
        img,
        processed,
        hough,
        prev_circle,
    )
    update_best(hough_on_thumb_detect(img, proc_for_hough, hough))
    # 3. 尝试稳健RANSAC检测
    update_best(robust_ransac_detect(processed))

    # # 4. 再次超时检查
    # if time.time() - t0 > TIME_BUDGET:
    #     best_circle, best_score = timeout_fallback_detection(
    #         processed_det, processed, hough, int(H), int(W)
    #     )
    #     if best_circle is not None:
    #         return best_circle, processed, best_score

    # 5. 尝试标准霍夫检测
    update_best(standard_hough_detect(img, proc_for_hough, hough))

    # 6. 尝试自适应霍夫检测
    if best_score < 15:
        update_best(adaptive_hough_detect(img, proc_for_hough, hough, brightness_mode))

    # —— 轮廓备选 —— #
    if best_score < 10:
        update_best(contour_detect(processed, hough))

    # —— padding-based fallback —— #
    # if time.time() - t0 > TIME_BUDGET:
    #     # 超时降级检测
    #     best_circle, best_score = timeout_fallback_detection(
    #         processed_det, processed, hough, int(H), int(W)
    #     )
    #     if best_circle is not None:
    #         return best_circle, processed, best_score

    if not best_circle or (best_score < 10) or touches_border(img, best_circle):
        update_best(padding_fallback_detect(processed, hough))

    if best_circle and not (hough.minRadius <= best_circle.radius <= hough.maxRadius):
        # —— 最终半径窗口一致性检查（严格遵守 UI 设定） —— #
        update_best(final_detect(processed, hough))
    if not best_circle:
        return None
    return DetectionResult(best_circle, best_score)
