import itertools
import logging
from pathlib import Path
import os, math, time

import cv2, numpy as np

from utils import (
    Hough,
    Position,
    Vector,
    imread_unicode,
    imwrite_unicode,
    get_memory_usage_mb,
    force_garbage_collection,
    SUPPORTED_EXTS,
)

from algorithms_circle import Circle, detect_circle_phd2_enhanced, masked_phase_corr
from version import VERSION


from algorithms_refine import refine_alignment_multi_roi
from numpy.typing import NDArray


# ------------------ 调试图保存 ------------------
def save_debug_image(
    processed_img: NDArray,
    target_center: Position[float],
    reference_center: Position[float],
    shift: Vector[float],
    confidence,
    debug_dir: Path,
    filename,
    reference_filename,
):
    try:
        if processed_img is None:
            return
        if processed_img.ndim == 2:
            debug_image = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        else:
            debug_image = processed_img.copy()
        cv2.circle(
            debug_image,
            (int(target_center.x), int(target_center.y)),
            5,
            (0, 0, 255),
            -1,
        )
        cv2.circle(
            debug_image,
            (int(reference_center.x), int(reference_center.y)),
            15,
            (0, 255, 255),
            3,
        )
        cv2.line(
            debug_image,
            (int(target_center.x), int(target_center.y)),
            (int(reference_center.x), int(reference_center.y)),
            (0, 255, 255),
            2,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        texts = [
            # f"Method: {method[:35]}",
            f"Shift: ({shift.x:.1f}, {shift.y:.1f})",
            f"Confidence: {confidence:.3f}",
            f"Reference: {reference_filename}",
            f"Mode: Incremental Processing",
        ]
        for j, t in enumerate(texts):
            cv2.putText(
                debug_image, t, (10, 25 + j * 25), font, 0.6, (255, 255, 255), 2
            )
        debug_path = debug_dir / f"debug_{filename}"
        imwrite_unicode(debug_path, debug_image)
    except Exception as e:
        print(f"调试图像生成失败: {e}")


# ------------------ 缩略图辅助 ------------------
def detect_circle_on_thumb(
    img: np.ndarray, hough: Hough, max_side=1600, strong_denoise=False
) -> tuple[Circle, float]:
    """
    Returns:
    - Circle: (cx, cy, radius) in original image scale
    - scale: float, the scale factor from original to thumbnail
    - quality: float, quality score of the detected circle
    - method: str, description of the detection method used
    Raises Exception if detection fails
    仅用于辅助选择参考图像
    """
    H, W = img.shape[:2]
    max_wh = max(H, W)
    scale = 1.0
    if max_wh > max_side:
        scale = max_side / float(max_wh)
    small = (
        cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else img
    )

    s_hough = Hough(
        minRadius=max(1, int(hough.minRadius * scale)),
        maxRadius=max(
            2, int(hough.minRadius * scale) + 1, int(hough.maxRadius * scale)
        ),
        param1=hough.param1,
        param2=hough.param2,
    )

    (
        circle_s,
        quality_s,
    ) = detect_circle_phd2_enhanced(small, s_hough, strong_denoise=strong_denoise)

    if circle_s is None:
        raise Exception("缩略图圆检测失败")

    circle = Circle(
        x=circle_s.x / scale,
        y=circle_s.y / scale,
        radius=circle_s.radius / scale,
    )

    return (
        circle,
        quality_s,
    )


# ------------------ 目录设置 ------------------
def _setup_directories(output_dir: Path) -> Path:
    """
    创建输出目录和调试目录

    Args:
        output_dir: 输出目录路径
        debug_mode: 是否启用调试模式

    Returns:
        tuple: (output_dir, debug_dir)

    Raises:
        Exception: 无法创建目录时抛出异常
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise Exception(f"无法创建输出文件夹: {output_dir}") from e

    # debug_dir = output_dir / "debug"
    # if debug_mode:
    #     try:
    #         debug_dir.mkdir(parents=True, exist_ok=True)
    #     except Exception as e:
    #         raise Exception(f"无法创建调试文件夹: {debug_dir}") from e

    return output_dir


# ------------------ 图像文件加载 ------------------
def _load_image_files(input_dir: Path) -> list[Path]:
    """
    加载输入目录中的图像文件

    Args:
        input_dir: 输入目录路径

    Returns:
        list[str]: 排序后的图像文件名列表

    Raises:
        Exception: 无法读取输入目录或没有找到支持的图像文件
    """
    try:
        image_files = sorted(
            itertools.chain.from_iterable(input_dir.glob(ext) for ext in SUPPORTED_EXTS)
        )
    except Exception as e:
        raise Exception(f"读取输入文件夹失败: {e}")

    if not image_files:
        raise Exception(f"在 '{input_dir}' 中未找到支持的图片文件")

    return image_files


# ------------------ 参考图像选择 ------------------
def _load_user_reference(
    reference_path: Path | None, hough: Hough, strong_denoise: bool = False
) -> tuple[NDArray, Circle, Path, float]:
    """
    加载用户指定的参考图像

    Args:
        reference_image_path: 参考图像路径
        hough: 霍夫变换参数
        strong_denoise: 是否使用强去噪

    Returns:
        tuple: (reference_image, reference_circle, reference_filename, best_quality)
    """
    reference_image = None
    reference_circle = None
    best_quality = 0.0

    if not (reference_path and os.path.exists(reference_path)):
        raise Exception("用户指定的参考图像路径无效或不存在")

    logging.info(f"加载用户指定的参考图像: {reference_path.name}")

    t_ref0 = time.time()
    ref_img = imread_unicode(Path(reference_path), cv2.IMREAD_UNCHANGED)
    if ref_img is None:
        raise Exception("参考图像读取失败")

    H, W = ref_img.shape[:2]
    logging.info(f"参考图尺寸: {W}x{H}")

    # 先在缩略图做，映射回原图
    try:
        circle, q = detect_circle_on_thumb(
            ref_img, hough, max_side=1600, strong_denoise=strong_denoise
        )

        reference_image = ref_img.copy()
        reference_circle = circle
        best_quality = q
    except Exception as e:
        logging.warning("缩略图检测失败，回退到原图做一次圆检测（可能较慢）...")
        t1 = time.time()
        circle_full, qf = detect_circle_phd2_enhanced(
            ref_img, hough, strong_denoise=strong_denoise
        )
        dt1 = time.time() - t1
        if circle_full is not None:
            reference_image = ref_img.copy()
            reference_circle = circle_full
            best_quality = float(qf)
        else:
            raise Exception("参考图像圆检测失败")

    return reference_image, reference_circle, reference_path, best_quality


def auto_select_reference(
    image_files: list[Path],
    hough: Hough,
    progress_callback=None,
    strong_denoise: bool = False,
) -> tuple[NDArray, Circle, Path, float]:
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
    reference_image = None
    reference_circle = None
    reference_path = None
    best_quality = 0.0

    scan_count = min(10, len(image_files))
    logging.info(f"自动选择参考图像 (扫描前{scan_count}张)...")

    for i, input_path in enumerate(image_files[:scan_count]):
        if progress_callback:
            progress_callback(int((i / scan_count) * 20), f"扫描参考图像: {input_path}")
        img0 = imread_unicode(input_path, cv2.IMREAD_UNCHANGED)
        if img0 is None:
            continue
        try:
            circle, q = detect_circle_on_thumb(
                img0, hough, max_side=1600, strong_denoise=strong_denoise
            )
            if q > best_quality:
                reference_image = img0.copy()
                reference_circle = circle
                reference_path = input_path
                best_quality = q
        except Exception as e:
            pass
        del img0
        force_garbage_collection()
    if reference_image is None or reference_circle is None or reference_path is None:
        raise Exception("未能自动选择参考图像")

    return reference_image, reference_circle, reference_path, best_quality


# ------------------ 单图像处理 ------------------
def _apply_initial_alignment(
    target_image: NDArray, circle: Circle, quality: float, reference_circle: Circle
) -> tuple[NDArray, Vector[float], float]:
    """
    应用初始圆心对齐

    Args:
        target_image: 目标图像
        circle: 目标图像的圆
        reference_circle: 参考图像的圆

    Returns:
        tuple: (aligned_image, shift, confidence)
    """
    shift = reference_circle - circle
    confidence = max(0.30, min(0.98, quality / 100.0))  # 使用默认质量值

    rows, cols = target_image.shape[:2]
    M = np.array([[1, 0, shift.x], [0, 1, shift.y]], dtype=np.float64)

    aligned = cv2.warpAffine(
        target_image,
        M,
        (cols, rows),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return aligned, shift, confidence


def _apply_refinement_alignment(
    aligned_image: NDArray,
    reference_image: NDArray,
    reference_circle: Circle,
    shift: Vector[float],
    use_advanced_alignment: bool = False,
    initial_confidence: float = 0.5,
) -> tuple[NDArray, float]:
    """
    应用精配准（多ROI或相位相关）

    Args:
        aligned_image: 已经初始对齐的图像
        reference_image: 参考图像
        reference_circle: 参考圆
        shift: 初始偏移量
        use_advanced_alignment: 是否使用高级多ROI对齐
        initial_confidence: 初始置信度

    Returns:
        tuple: (refined_image, confidence)
    """
    rows, cols = aligned_image.shape[:2]
    confidence = initial_confidence

    try:
        reference_radius = reference_circle.radius

        if reference_radius is None:
            return aligned_image, confidence

        ref_gray = (
            reference_image
            if reference_image.ndim == 2
            else cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        )
        tgt_gray2 = (
            aligned_image
            if aligned_image.ndim == 2
            else cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
        )

        if use_advanced_alignment:
            roi_size = max(64, min(160, int(reference_radius * 0.18)))
            max_refine_delta_px = 6.0
            t_refine = time.time()
            M2x3, score, n_inliers, theta_deg = refine_alignment_multi_roi(
                ref_gray,
                tgt_gray2,
                reference_circle,
                n_rois=16,
                roi_size=roi_size,
                search=12,
                use_phasecorr=True,
                use_ecc=False,
                base_shift=shift,
                max_refine_delta_px=max_refine_delta_px,
            )
            dt_refine = time.time() - t_refine

            roi_used = roi_size
            logging.info(
                f"    [Refine] score={score:.3f}, inliers={n_inliers}, roi_init≈{roi_used}, t={dt_refine:.2f}s"
            )
            residual = None
            if M2x3 is not None:
                tx = float(M2x3[0, 2])
                ty = float(M2x3[1, 2])
                residual = (tx**2 + ty**2) ** 0.5
                logging.info(f"    [Refine] 残差=Δ{residual:.2f}px")
                if residual > max_refine_delta_px:
                    M2x3 = None
                    logging.info(
                        f"    [Refine] 残差过大(Δ={residual:.2f}px > {max_refine_delta_px:.1f}px)，放弃精配准并保持霍夫平移"
                    )
            if M2x3 is not None:
                aligned_image = cv2.warpAffine(
                    aligned_image,
                    M2x3,
                    (cols, rows),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                confidence = max(confidence, float(score))
                logging.info(
                    f"Multi-ROI refine (仅平移, inliers={n_inliers}, roi_init≈{roi_used}, Δ={residual:.2f}px, gate≤{max_refine_delta_px:.0f}px, {dt_refine:.2f}s)"
                )

            else:
                logging.info("    [Refine] 无有效解，回退 Masked PhaseCorr")
                # 遮罩相位相关微调（仅平移）
                t_pc = time.time()
                d2 = masked_phase_corr(
                    ref_gray,
                    tgt_gray2,
                    reference_circle,
                )
                dt_pc = time.time() - t_pc
                if abs(d2.x) > 1e-3 or abs(d2.y) > 1e-3:
                    M2 = np.array([[1, 0, d2.x], [0, 1, d2.y]], dtype=np.float32)
                    aligned_image = cv2.warpAffine(
                        aligned_image,
                        M2,
                        (cols, rows),
                        flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    logging.info(f"Masked PhaseCorr ({dt_pc:.2f}s)")

        else:
            # 未启用高级：遮罩相位相关微调
            d2 = masked_phase_corr(
                ref_gray,
                tgt_gray2,
                reference_circle,
            )
            if abs(d2.x) > 1e-3 or abs(d2.y) > 1e-3:
                M2 = np.array([[1, 0, d2.x], [0, 1, d2.y]], dtype=np.float32)
                aligned_image = cv2.warpAffine(
                    aligned_image,
                    M2,
                    (cols, rows),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                logging.info(f"Masked PhaseCorr")

    except Exception as e:
        logging.warning(f"    [Refine异常] {e}")

    return aligned_image, confidence


def _process_single_image(
    filename: str,
    input_dir: Path,
    output_dir: Path,
    reference_image: NDArray,
    reference_circle: Circle,
    reference_path: Path,
    hough: Hough,
    last_circle: Circle | None,
    use_advanced_alignment: bool = False,
    strong_denoise: bool = False,
) -> tuple[bool, Circle | None, dict, dict]:
    """
    处理单个图像的对齐

    Args:
        filename: 文件名
        input_dir: 输入目录
        output_dir: 输出目录
        reference_image: 参考图像
        reference_circle: 参考圆
        reference_filename: 参考文件名
        hough: 霍夫变换参数
        last_circle: 上一帧的圆（用于先验）
        debug_mode: 调试模式
        debug_image_basename: 调试图像基准名
        use_advanced_alignment: 是否使用高级对齐
        strong_denoise: 是否使用强去噪
        debug_dir: 调试目录

    Returns:
        tuple: (success, new_last_circle)
    """
    brightness_stats = {"bright": 0, "normal": 0, "dark": 0}
    method_stats = {}

    try:
        input_path = input_dir / filename

        # 参考图：直接另存
        if filename == reference_path:
            output_path = output_dir / f"aligned_{filename}"
            if imwrite_unicode(output_path, reference_image):
                logging.info(f"  🎯 {filename}: [参考图像] 已保存")
                return True, last_circle, brightness_stats, method_stats
            else:
                logging.info(f"  ✗ {filename}: 保存失败")
                return False, last_circle, brightness_stats, method_stats

        # 读取目标
        t_read = time.time()
        target_image: NDArray | None = imread_unicode(input_path, cv2.IMREAD_UNCHANGED)
        if target_image is None:
            logging.info(f"  ✗ {filename}: 读取失败")
            return False, last_circle, brightness_stats, method_stats
        dt_read = time.time() - t_read

        # 圆检测
        t_det = time.time()
        circle, quality = detect_circle_phd2_enhanced(
            target_image,
            hough,
            strong_denoise=strong_denoise,
            prev_circle=last_circle,
        )
        dt_det = time.time() - t_det

        if circle is None:
            logging.info(f"  ✗ {filename}: 圆检测失败(耗时 {dt_det:.2f}s)")
            del target_image
            return False, last_circle, brightness_stats, method_stats

        # 初始对齐
        aligned, shift, confidence = _apply_initial_alignment(
            target_image, circle, quality, reference_circle
        )

        # 精配准
        aligned, confidence = _apply_refinement_alignment(
            aligned,
            reference_image,
            reference_circle,
            shift,
            use_advanced_alignment,
            confidence,
        )

        # 保存
        out_path = output_dir / f"aligned_{filename}"
        if imwrite_unicode(out_path, aligned):
            # 更新上一帧先验
            new_last_circle = circle
            logging.info(
                f"  ✓ {filename}: 偏移=({shift.x:.1f},{shift.y:.1f}), "
                f"质量={quality:.1f}, 置信度={confidence:.3f}, 圆检耗时={dt_det:.2f}s, 读取={dt_read:.2f}s"
            )

            del target_image, aligned

            force_garbage_collection()
            return True, new_last_circle, brightness_stats, method_stats
        else:
            logging.info(f"  ✗ {filename}: 变换成功但保存失败")
            del target_image, aligned

            force_garbage_collection()
            return False, last_circle, brightness_stats, method_stats

    except Exception as e:
        logging.info(f"  ✗ {filename}: 处理异常 - {e}")
        for v in ["target_image", "aligned", "processed"]:
            if v in locals():
                del locals()[v]
        force_garbage_collection()
        return False, last_circle, brightness_stats, method_stats


# ------------------ 统计记录 ------------------
def _log_processing_stats(
    success_count: int,
    input_files_num: int,
    failed_files: list[Path],
    reference_path: Path,
    use_advanced_alignment: bool,
    brightness_stats: dict,
    method_stats: dict,
) -> None:
    """
    记录处理统计信息

    Args:
        success_count: 成功数量
        input_files_num: 输入文件总数
        failed_files: 失败文件列表
        reference_filename: 参考文件名
        use_advanced_alignment: 是否使用高级对齐
        brightness_stats: 亮度统计
        method_stats: 方法统计
    """
    logging.info(f"增量对齐完成! 成功对齐 {success_count}/{input_files_num} 张图像")
    logging.info(f"使用参考图像: {reference_path}")
    logging.info(
        f"对齐算法: {'多ROI精配准（仅平移）' if use_advanced_alignment else 'PHD2圆心算法'}"
    )
    if failed_files:
        failed_file_names = [f.name for f in failed_files[:5]]
        head = ", ".join(failed_file_names) + ("..." if len(failed_files) > 5 else "")
        logging.info(f"失败文件({len(failed_files)}): {head}")
    if method_stats:
        logging.info(
            "圆检测方法统计: "
            + ", ".join([f"{k}={v}" for k, v in method_stats.items()])
        )
    logging.info(f"当前内存使用: {get_memory_usage_mb():.1f} MB")


# ------------------ 主流程 ------------------


def align_moon_images_incremental(
    input_dir: Path,
    output_dir: Path,
    hough: Hough,
    completion_callback=None,
    progress_callback=None,
    reference_path: Path | None = None,
    use_advanced_alignment=False,
    strong_denoise=False,
):
    """
    月食图像增量对齐主函数 - 重构后的协调器版本

    将原本的单体大函数拆分为多个职责明确的小函数，提高可维护性和可测试性
    """
    try:
        # 1. 设置目录
        output_dir = _setup_directories(output_dir)

        # 2. 加载图像文件
        image_files = _load_image_files(input_dir)
        input_files_num = len(image_files)

        # 记录基本信息
        logging.info(f"月食圆面对齐工具 V{VERSION} - 增量处理版")
        logging.info(f"处理模式: 增量处理 (边检测边保存)")
        logging.info(f"文件总数: {input_files_num}")
        logging.info(f"多ROI精配准: {'启用' if use_advanced_alignment else '禁用'}")

        # 3. 选择参考图像
        logging.info("阶段 1/2: 确定参考图像...")
        try:
            reference_image, reference_circle, reference_path, best_quality = (
                _load_user_reference(reference_path, hough, strong_denoise)
            )

        except Exception as e:
            logging.warning(f"用户指定参考图像无效: {str(e)}，将自动选择")
            # 如果用户指定失败，自动选择
            try:
                reference_image, reference_circle, reference_path, best_quality = (
                    auto_select_reference(
                        image_files,
                        hough,
                        progress_callback,
                        strong_denoise,
                    )
                )
            except Exception as e:
                raise Exception(
                    "无法找到有效的参考图像，请检查图像质量和参数设置"
                ) from e

        logging.info(
            f"🎯 最终参考图像: {reference_path.name}, 质量评分={best_quality:.1f}"
        )

        # 4. 处理所有图像
        logging.info(f"\n阶段 2/2: 增量处理所有图像...")
        success_count = 0
        failed_files = []
        total_brightness_stats = {"bright": 0, "normal": 0, "dark": 0}
        total_method_stats = {}

        # 以参考图圆作为先验，后续逐帧更新
        last_circle: Circle | None = reference_circle

        for i, image_file in enumerate(image_files):
            filename = image_file.name
            if progress_callback:
                progress_callback(
                    20 + int((i / input_files_num) * 80), f"处理: {filename}"
                )

            # 处理单个图像
            success, new_last_circle, brightness_stats, method_stats = (
                _process_single_image(
                    filename=filename,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    reference_image=reference_image,
                    reference_circle=reference_circle,
                    reference_path=reference_path,
                    hough=hough,
                    last_circle=last_circle,
                    use_advanced_alignment=use_advanced_alignment,
                    strong_denoise=strong_denoise,
                )
            )

            if success:
                success_count += 1
                last_circle = new_last_circle
            else:
                failed_files.append(filename)

            # 累计统计信息
            for k, v in brightness_stats.items():
                total_brightness_stats[k] += v
            for k, v in method_stats.items():
                total_method_stats[k] = total_method_stats.get(k, 0) + v

        # 5. 完成处理
        if progress_callback:
            progress_callback(100, "处理完成")

        del reference_image
        force_garbage_collection()

        # 记录统计信息
        _log_processing_stats(
            success_count=success_count,
            input_files_num=input_files_num,
            failed_files=failed_files,
            reference_path=reference_path,
            use_advanced_alignment=use_advanced_alignment,
            brightness_stats=total_brightness_stats,
            method_stats=total_method_stats,
        )

        if completion_callback:
            completion_callback(
                True, f"增量处理完成！成功对齐 {success_count}/{input_files_num} 张图像"
            )

    except Exception as e:
        import traceback

        err = f"增量处理过程中发生错误: {e}\n{traceback.format_exc()}"
        logging.error(err)
        if completion_callback:
            completion_callback(False, err)
    finally:
        force_garbage_collection()
