import itertools
import logging
from pathlib import Path
import os, math, time
from typing import Iterable

import cv2, numpy as np

from image import Image, ImageFile
from utils import (
    MAX_SCAN_COUNT,
    DetectionResult,
    Hough,
    Position,
    Vector,
    get_memory_usage_mb,
)

from circle_detection import Circle, detect_circle, masked_phase_corr
from version import VERSION


from algorithms_refine import refine_alignment_multi_roi
from numpy.typing import NDArray


# ------------------ 调试图保存 ------------------
# def save_debug_image(
#     processed_img: NDArray,
#     target_center: Position[float],
#     reference_center: Position[float],
#     shift: Vector[float],
#     confidence,
#     debug_dir: Path,
#     filename,
#     reference_filename,
# ):
#     try:
#         if processed_img is None:
#             return
#         if processed_img.ndim == 2:
#             debug_image = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
#         else:
#             debug_image = processed_img.copy()
#         cv2.circle(
#             debug_image,
#             (int(target_center.x), int(target_center.y)),
#             5,
#             (0, 0, 255),
#             -1,
#         )
#         cv2.circle(
#             debug_image,
#             (int(reference_center.x), int(reference_center.y)),
#             15,
#             (0, 255, 255),
#             3,
#         )
#         cv2.line(
#             debug_image,
#             (int(target_center.x), int(target_center.y)),
#             (int(reference_center.x), int(reference_center.y)),
#             (0, 255, 255),
#             2,
#         )
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         texts = [
#             # f"Method: {method[:35]}",
#             f"Shift: ({shift.x:.1f}, {shift.y:.1f})",
#             f"Confidence: {confidence:.3f}",
#             f"Reference: {reference_filename}",
#             f"Mode: Incremental Processing",
#         ]
#         for j, t in enumerate(texts):
#             cv2.putText(
#                 debug_image, t, (10, 25 + j * 25), font, 0.6, (255, 255, 255), 2
#             )
#         debug_path = debug_dir / f"debug_{filename}"
#         imwrite_unicode(debug_path, debug_image)
#     except Exception as e:
#         print(f"调试图像生成失败: {e}")


# ------------------ 缩略图辅助 ------------------
def detect_circle_on_thumb(
    bgr: NDArray, hough: Hough, max_side=1600, strong_denoise=False
) -> DetectionResult | None:
    """
    Returns:
    - Circle: (cx, cy, radius) in original image scale
    - scale: float, the scale factor from original to thumbnail
    - quality: float, quality score of the detected circle
    - method: str, description of the detection method used
    Raises Exception if detection fails
    仅用于辅助选择参考图像
    """
    H, W = bgr.shape[:2]
    max_wh = max(H, W)
    scale = 1.0
    if max_wh > max_side:
        scale = max_side / float(max_wh)
    small = (
        cv2.resize(bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else bgr
    )

    s_hough = Hough(
        minRadius=max(1, int(hough.minRadius * scale)),
        maxRadius=max(
            2, int(hough.minRadius * scale) + 1, int(hough.maxRadius * scale)
        ),
        param1=hough.param1,
        param2=hough.param2,
    )

    result = detect_circle(small, s_hough, strong_denoise=strong_denoise)
    if result is None:
        logging.error("缩略图圆检测失败")
        return None

    circle = Circle(
        x=result.circle.x / scale,
        y=result.circle.y / scale,
        radius=result.circle.radius / scale,
    )

    return DetectionResult(circle, result.quality)


# ------------------ 参考图像选择 ------------------
def _load_user_reference(
    reference_file: ImageFile, hough: Hough, strong_denoise: bool = False
) -> DetectionResult | None:

    ref_img = reference_file.image
    if ref_img is None:
        raise Exception(f"无法加载参考图像: {reference_file.path}")

    ref_bgr = ref_img.bgr

    H, W = ref_bgr.shape[:2]
    logging.info(f"参考图尺寸: {W}x{H}")

    # 先在缩略图做，映射回原图
    try:
        return detect_circle_on_thumb(
            ref_bgr, hough, max_side=1600, strong_denoise=strong_denoise
        )
    except Exception:
        pass

    logging.warning("缩略图检测失败，回退到原图做一次圆检测（可能较慢）...")

    result = detect_circle(ref_bgr, hough, strong_denoise=strong_denoise)

    if not result:
        logging.warning(f"用户指定参考图像无效: {reference_file.path.name}，将自动选择")
    return result


def auto_select_reference(
    input_files: Iterable[ImageFile],
    hough: Hough,
    strong_denoise: bool = False,
) -> tuple[Circle, ImageFile] | None:
    """
    自动选择参考图像（扫描前N张质量最好的）

    Args:
        image_files: 图像文件列表
        input_dir: 输入目录路径
        hough: 霍夫变换参数
        progress_callback: 进度回调函数
        strong_denoise: 是否使用强去噪

    Returns:
        tuple: (reference_image, reference_circle, reference_filename, best_quality)
    """

    best_result: DetectionResult | None = None
    reference_file: ImageFile | None = None

    logging.info(f"自动选择参考图像 (扫描前{MAX_SCAN_COUNT}张)...")

    for i, input_file in enumerate(input_files):
        if i >= MAX_SCAN_COUNT:
            break
        img = input_file.image
        if not img:
            continue

        result = detect_circle_on_thumb(
            img.bgr, hough, max_side=1600, strong_denoise=strong_denoise
        )
        if not result:
            continue
        if not best_result:
            best_result = result
            reference_file = input_file
        elif result.quality > best_result.quality:
            best_result = result
            reference_file = input_file
    if not best_result or not reference_file:
        return None
    logging.info(
        f"🎯 最终参考图像: {reference_file.path.name}, 质量评分={best_result.quality:.1f}"
    )
    return best_result.circle, reference_file


def get_reference(
    reference_path: Path | None,
    input_files: dict[Path, ImageFile],
    hough: Hough,
    strong_denoise: bool,
) -> tuple[Circle, ImageFile] | None:
    logging.info("阶段 1/2: 确定参考图像...")
    if reference_path and reference_path in input_files:
        reference_file = input_files[reference_path]
        result = _load_user_reference(reference_file, hough, strong_denoise)
        if result:
            return result.circle, reference_file
    return auto_select_reference(
        input_files.values(),
        hough,
        strong_denoise,
    )


# ------------------ 单图像处理 ------------------
def initial_align(img: Image, shift: Vector[float]) -> Image:
    """
    应用初始圆心对齐

    Args:
        bgr: 目标图像
        circle: 目标图像的圆
        reference_circle: 参考图像的圆

    Returns:
        tuple: (aligned_image, shift, confidence)
    """

    # 使用默认质量值

    M = np.array([[1, 0, shift.x], [0, 1, shift.y]], dtype=np.float64)

    aligned = cv2.warpAffine(
        img.rgb,
        M,
        img.col_row,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    logging.info(f"初始对齐: shift=({shift.x:.1f},{shift.y:.1f})")
    return Image(rgb=aligned)


def advanced_align(
    img: Image, reference_img: Image, reference_circle: Circle, shift: Vector[float]
) -> Image | None:
    roi_size = max(64, min(160, int(reference_circle.radius * 0.18)))
    max_refine_delta_px = 6.0
    t_refine = time.time()
    M2x3, score, n_inliers = refine_alignment_multi_roi(
        reference_img.gray,
        img.gray,
        reference_circle,
        n_rois=16,
        roi_size=roi_size,
        search=12,
        use_phasecorr=True,
        base_shift=shift,
        max_refine_delta_px=max_refine_delta_px,
    )
    dt_refine = time.time() - t_refine

    roi_used = roi_size
    logging.info(
        f"    [Refine] score={score:.3f}, inliers={n_inliers}, roi_init≈{roi_used}, t={dt_refine:.2f}s"
    )

    t = Vector(float(M2x3[0, 2]), float(M2x3[1, 2]))

    residual = t.norm()
    logging.info(f"    [Refine] 残差=Δ{residual:.2f}px")
    if residual > max_refine_delta_px:
        logging.warning(
            f"    [Refine] 残差过大(Δ={residual:.2f}px > {max_refine_delta_px:.1f}px)，放弃精配准并保持霍夫平移"
        )
        return None

    logging.info(
        f"Multi-ROI refine (仅平移, inliers={n_inliers}, roi_init≈{roi_used}, Δ={residual:.2f}px, gate≤{max_refine_delta_px:.0f}px, {dt_refine:.2f}s)"
    )
    aliged_rbg = cv2.warpAffine(
        img.rgb,
        M2x3,
        img.col_row,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return Image(rgb=aliged_rbg)


def mask_phase_align(
    img: Image,
    reference_img: Image,
    reference_circle: Circle,
) -> Image | None:

    # 未启用高级：遮罩相位相关微调
    shift = masked_phase_corr(
        reference_img.gray,
        img.gray,
        reference_circle,
    )
    if abs(shift.x) <= 1e-3 and abs(shift.y) <= 1e-3:
        return None

    M2 = np.array([[1, 0, shift.x], [0, 1, shift.y]], dtype=np.float32)
    aligned_rgb = cv2.warpAffine(
        img.rgb,
        M2,
        img.col_row,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    logging.info(f"Masked PhaseCorr")

    return Image(rgb=aligned_rgb)


def align(
    img: Image,
    circle: Circle,
    reference_img: Image,
    reference_circle: Circle,
    use_advanced_alignment: bool,
) -> Image:
    shift = reference_circle - circle
    initial_aligned = initial_align(img, shift)

    if use_advanced_alignment:
        aligned = advanced_align(
            img,
            reference_img,
            reference_circle,
            shift,
        )
        if aligned:
            return aligned

    return (
        mask_phase_align(initial_aligned, reference_img, reference_circle)
        or initial_aligned
    )


def process_single_image(
    input_file: ImageFile,
    output_dir: Path,
    ref_image: Image,
    ref_circle: Circle,
    hough: Hough,
    last_circle: Circle | None,
    use_advanced_alignment: bool = False,
    strong_denoise: bool = False,
):

    t_det = time.time()
    input_image = input_file.image
    if input_image is None:
        return None
    input_bgr = input_image.bgr
    result = detect_circle(
        input_bgr,
        hough,
        strong_denoise=strong_denoise,
        prev_circle=last_circle,
    )
    dt_det = time.time() - t_det

    if result is None:
        logging.error(f"  ✗ {input_file.path.name}: 圆检测失败(耗时 {dt_det:.2f}s)")
        return None
    else:
        logging.info(
            f"  ○ {input_file.path.name}: 圆检测成功 (质量={result.quality:.1f}, 半径={result.circle.radius:.1f}px, 耗时 {dt_det:.2f}s)"
        )

    output_image = align(
        input_image, result.circle, ref_image, ref_circle, use_advanced_alignment
    )

    output_image.exif = input_image.exif
    output_image.icc = input_image.icc
    output_file = ImageFile(output_dir / f"{input_file.path.name}", mode="w")

    # 保存
    output_file.image = output_image
    output_file.save()

    return result.circle


# ------------------ 主流程 ------------------


def process_images(
    input_dir: Path,
    output_dir: Path,
    hough: Hough,
    reference_path: Path | None = None,
    use_advanced_alignment=False,
    strong_denoise=False,
):
    """
    月食图像增量对齐主函数 - 重构后的协调器版本

    将原本的单体大函数拆分为多个职责明确的小函数，提高可维护性和可测试性
    """

    # 1. 设置目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 加载图像文件
    input_files = ImageFile.load(input_dir)

    # 记录基本信息
    logging.info(f"月食圆面对齐工具 V{VERSION} - 增量处理版")
    logging.info(f"处理模式: 增量处理 (边检测边保存)")
    logging.info(f"文件总数: {len(input_files)}")
    logging.info(f"多ROI精配准: {'启用' if use_advanced_alignment else '禁用'}")

    # 3. 选择参考图像
    if not (rt := get_reference(reference_path, input_files, hough, strong_denoise)):
        logging.error("未能确定参考图像，处理终止")
        return
    ref_circle, ref_file = rt
    assert ref_file.image
    logging.info(
        f"🎯 参考图像: {ref_file.path.name}, 圆心=({ref_circle.x:.1f},{ref_circle.y:.1f}), 半径={ref_circle.radius:.1f}px"
    )

    # 4. 处理所有图像
    logging.info(f"\n阶段 2/2: 顺序处理所有图像...")
    success_count = 0
    failed_files: list[ImageFile] = []

    # 以参考图圆作为先验，后续逐帧更新
    last_circle: Circle | None = ref_circle

    for input_file in input_files.values():

        # 处理单个图像
        new_last_circle = process_single_image(
            input_file=input_file,
            output_dir=output_dir,
            ref_image=ref_file.image,
            ref_circle=ref_circle,
            hough=hough,
            last_circle=last_circle,
            use_advanced_alignment=use_advanced_alignment,
            strong_denoise=strong_denoise,
        )

        if new_last_circle:
            success_count += 1
            last_circle = new_last_circle
        else:
            failed_files.append(input_file)

    logging.info(f"增量对齐完成! 成功对齐 {success_count}/{len(input_files)} 张图像")

    logging.info(
        f"对齐算法: {'多ROI精配准（仅平移）' if use_advanced_alignment else 'PHD2圆心算法'}"
    )
    if failed_files:
        failed_file_names = [f.path.name for f in failed_files[:5]]
        head = ", ".join(failed_file_names) + ("..." if len(failed_files) > 5 else "")
        logging.info(f"失败文件({len(failed_files)}): {head}")

    logging.debug(f"当前内存使用: {get_memory_usage_mb():.1f} MB")
