from enum import Enum

import logging


import numpy as np
import cv2
from numpy._typing._array_like import NDArray

from image import Image
from utils import (
    Circle,
    DetectionResult,
    HoughParams,
    PointArray,
    Vector,
    VectorArray,
    ring_mask,
    soft_disk_mask,
)
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
    inners: PointArray = directions * (circle.radius - 2) + circle.center
    outers: PointArray = directions * (circle.radius + 2) + circle.center
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
) -> PointArray | None:
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return None

    pts = PointArray(np.stack([xs, ys], axis=1), safe=False)
    if prev_circle is None:
        return pts
    vectors = pts - prev_circle.center
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


def initial_process(
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


def hough_on_thumb_detect(masked_gray: NDArray, params: HoughParams) -> list[Circle]:
    height, width = masked_gray.shape[:2]
    max_side = max(height, width)
    circles = []
    if max_side <= 1800:
        return []
    scale = 1800.0 / max_side
    small_height, small_width = int(height * scale), int(width * scale)
    small = cv2.resize(
        masked_gray,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )
    params = HoughParams(
        minRadius=max(1, int(params.minRadius * scale)),
        maxRadius=max(2, int(params.maxRadius * scale)),
        param1=params.param1,
        param2=max(params.param2 - 5, 10),
    )

    circles = cv2.HoughCircles(
        small,
        minDist=small_height // 2,
        **params._asdict(),
    )

    circles = [Circle.from_ndarray(c / scale) for c in circles[0]]
    logging.info("缩放霍夫(thumb)")
    return circles


# ------------------ ROI构建 ------------------
def build_detection_roi(
    processed: NDArray,
    params: HoughParams,
    prev_circle: Circle | None,
) -> NDArray:
    height, width = processed.shape[:2]
    # —— 粗估中心半径，构建环形 ROI —— #
    est: Circle | None = rough_center_radius(
        processed, params.minRadius, params.maxRadius
    )
    if est:
        ring = ring_mask(width, height, est, inner=0.70, outer=1.15) * np.uint8(255)
        processed = cv2.bitwise_and(processed, processed, mask=ring)

    # —— 若给出上一帧圆心半径，合并一个"历史先验"环形 ROI —— #
    if prev_circle:

        inner = max(
            0.70,
            min(0.85, (params.minRadius / max(prev_circle.radius, 1e-6)) * 0.9),
        )
        outer = min(
            1.30,
            max(1.15, (params.maxRadius / max(prev_circle.radius, 1e-6)) * 1.05),
        )
        ring_prev = ring_mask(
            width, height, prev_circle, inner=inner, outer=outer
        ) * np.uint8(255)
        processed = cv2.bitwise_and(processed, processed, mask=ring_prev)
        return processed

    return processed


# ------------------ 超时检测函数 ------------------
def timeout_fallback_detection(gray: NDArray, params: HoughParams) -> list[Circle]:
    height, width = gray.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    small_height, small_width = int(height * scale), int(width * scale)
    small = cv2.resize(
        gray,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )
    params = HoughParams(
        minRadius=max(1, int(params.minRadius * scale)),
        maxRadius=max(2, int(params.maxRadius * scale)),
        param1=max(params.param1, 20),
        param2=max(params.param2 - 5, 8),
    )

    circles = cv2.HoughCircles(
        small,
        minDist=small_height // 2,
        **params._asdict(),
    )

    circles = [Circle.from_ndarray(c) for c in circles[0]]
    if circles:
        logging.info("超时降级(thumb)")

    return circles


# ------------------ 稳健RANSAC检测 ------------------


def detect_circle_robust(
    gray: NDArray, prev_circle: Circle | None = None
) -> list[Circle]:
    pts = edge_points_outer_rim(gray, prev_circle)
    if not pts:
        return []
    model, inliers = ransac(
        data=pts._arr,
        model_class=CircleModel,
        min_samples=3,
        residual_threshold=2.0,
        max_trials=120,
        stop_probability=0.99,  # type: ignore
    )
    if np.sum(inliers) < 40:  # type: ignore
        return []
    cy, cx, r = model.params  # type: ignore
    cand = Circle(float(cx), float(cy), float(r))

    vectors = pts - cand.center
    arctans = np.arctan2(vectors.y, vectors.x)
    span: np.float64 = np.ptp(arctans)
    if prev_circle and span < (2 * np.pi / 3.0):  # <120°
        cand = Circle(cand.x, cand.y, prev_circle.radius)
    return [cand]


# ------------------ 标准霍夫检测 ------------------
def standard_hough_detect(masked_gray: NDArray, hough: HoughParams) -> list[Circle]:

    height, width = masked_gray.shape[:2]
    circles = cv2.HoughCircles(
        masked_gray,
        minDist=height // 2,
        **hough._asdict(),
    )
    if circles is None:
        return []

    circles = [Circle(*c) for c in circles[0]]
    if circles:
        logging.info("标准霍夫")
    return circles


# ------------------ 自适应霍夫检测 ------------------
def adaptive_hough_detect(
    masked_gray: NDArray,
    hough: HoughParams,
    brightness_mode: BrightnessMode,
) -> list[Circle]:

    height, width = masked_gray.shape[:2]
    if brightness_mode == BrightnessMode.BRIGHT:
        params = HoughParams(
            minRadius=hough.minRadius,
            maxRadius=hough.maxRadius,
            param1=hough.param1 + 20,
            param2=max(hough.param2 - 5, 10),
        )
    elif brightness_mode == BrightnessMode.DARK:
        params = HoughParams(
            minRadius=hough.minRadius,
            maxRadius=hough.maxRadius,
            param1=max(hough.param1 - 15, 20),
            param2=max(hough.param2 - 10, 5),
        )
    else:
        params = HoughParams(
            minRadius=hough.minRadius,
            maxRadius=hough.maxRadius,
            param1=hough.param1,
            param2=max(hough.param2 - 8, 8),
        )

    circles = cv2.HoughCircles(
        masked_gray,
        minDist=height // 2,
        **params._asdict(),
    )
    if circles is None:
        return []
    circles = [Circle(*c) for c in circles[0]]
    if circles:
        logging.info(f"自适应霍夫(P1={params.param1},P2={params.param2})")

    return circles


# ------------------ 轮廓检测 ------------------
def contour_detect(gray: NDArray, hough: HoughParams) -> list[Circle]:
    mean_val = float(np.mean(gray))
    tv = max(50, int(mean_val * 0.7))
    _, binary = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
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
    masked_gray: NDArray,
    params: HoughParams,
) -> list[Circle]:

    pad = int(max(32, round(params.maxRadius * 1.2)))
    padded_gray = cv2.copyMakeBorder(
        masked_gray,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    # Use constant black padding to avoid mirrored ghosts influencing Hough
    est = rough_center_radius(
        padded_gray, int(params.minRadius * 1.1), int(params.maxRadius * 1.1)
    )
    padded_masked_gray = padded_gray
    if est is not None:
        ring = ring_mask(
            padded_gray.shape[0],
            padded_gray.shape[1],
            est,
            inner=0.70,
            outer=1.15,
        ) * np.uint8(255)
        padded_masked_gray = cv2.bitwise_and(padded_gray, padded_gray, mask=ring)

    height, width = padded_masked_gray.shape[:2]
    params = HoughParams(
        minRadius=int(max(1, params.minRadius * 1.1)),
        maxRadius=int(max(2, params.maxRadius * 1.1)),
        param1=max(params.param1, 20),
        param2=max(params.param2 - 5, 8),
    )
    circles = cv2.HoughCircles(
        padded_masked_gray,
        minDist=height // 2,
        **params._asdict(),
    )
    circles = [Circle.from_ndarray(c) for c in circles[0]]
    return circles or detect_circle_robust(padded_gray)


# ------------------ 最终圆验证 ------------------
def final_detect(
    gray: NDArray,
    params: HoughParams,
) -> list[Circle]:

    # 在严格窗口内再做一次快速霍夫重试
    height, width = gray.shape

    params = HoughParams(
        minRadius=max(1, params.minRadius),
        maxRadius=params.maxRadius,
        param1=max(params.param1, 20),
        param2=max(params.param2 - 5, 8),
    )
    circles = cv2.HoughCircles(
        gray,
        minDist=max(16, min(height, width) // 4),
        **params._asdict(),
    )

    circles = [Circle.from_ndarray(c) for c in circles[0]]
    circles = [c for c in circles if params.minRadius <= c.radius <= params.maxRadius]

    return circles


def detect_circle(
    img: Image,
    params: HoughParams,
    strong_denoise=False,
    prev_circle: Circle | None = None,
) -> DetectionResult | None:
    """
    主检测函数 - 重构后的协调器版本

    将原本的单体大函数拆分为多个职责明确的辅助函数
    """

    best_result: DetectionResult | None = None

    # 1. 设置检测环境
    gray, brightness_mode = initial_process(img, strong_denoise)

    def update_best(circles: list[Circle]):
        nonlocal best_result, gray
        for c in circles:
            quality = evaluate_circle_quality(img, c)
            if not best_result:
                best_result = DetectionResult(c, quality)
            elif quality > best_result.quality:
                best_result = DetectionResult(c, quality)

    # 2. 构建检测ROI
    masked_gray = build_detection_roi(
        gray,
        params,
        prev_circle,
    )
    update_best(hough_on_thumb_detect(masked_gray, params))
    # 3. 尝试稳健RANSAC检测
    update_best(detect_circle_robust(gray))

    # # 4. 再次超时检查
    # if time.time() - t0 > TIME_BUDGET:
    #     best_circle, best_score = timeout_fallback_detection(
    #         processed_det, processed, hough, int(H), int(W)
    #     )
    #     if best_circle is not None:
    #         return best_circle, processed, best_score

    # 5. 尝试标准霍夫检测
    update_best(standard_hough_detect(masked_gray, params))

    # 6. 尝试自适应霍夫检测
    if not best_result or best_result.quality < 15:
        update_best(adaptive_hough_detect(masked_gray, params, brightness_mode))

    # —— 轮廓备选 —— #
    if not best_result or best_result.quality < 10:
        update_best(contour_detect(gray, params))

    # —— padding-based fallback —— #
    # if time.time() - t0 > TIME_BUDGET:
    #     # 超时降级检测
    #     best_circle, best_score = timeout_fallback_detection(
    #         processed_det, processed, hough, int(H), int(W)
    #     )
    #     if best_circle is not None:
    #         return best_circle, processed, best_score

    if (
        not best_result
        or (best_result.quality < 10)
        or touches_border(img.width, img.height, best_result.circle)
    ):
        update_best(padding_fallback_detect(gray, params))

    if best_result and not (
        params.minRadius <= best_result.circle.radius <= params.maxRadius
    ):
        # —— 最终半径窗口一致性检查（严格遵守 UI 设定） —— #
        update_best(final_detect(gray, params))

    return best_result
