import itertools
from pathlib import Path
import os, math, time
from typing import Literal, NamedTuple
import cv2, numpy as np

from utils import (
    Hough,
    log, 
    imread_unicode, imwrite_unicode, get_memory_usage_mb,
    force_garbage_collection, MemoryManager, SUPPORTED_EXTS
)

from algorithms_circle import Circle, detect_circle_phd2_enhanced, masked_phase_corr
from version import VERSION
# refine 返回可能是 (M, score, nin) 也可能是 (M, theta_deg, score, nin)
from algorithms_refine import refine_alignment_multi_roi  # 兼容旧/新签名



# 兼容不同版本 refine 返回值
# 可能: (M, score, nin) / (M, theta_deg, score, nin) / (M, score, nin, theta_deg)
# 返回统一: (M, theta_deg, score, nin)
def _unpack_refine_result(res):
    M = None; theta_deg = 0.0; score = 0.0; nin = 0
    if not isinstance(res, tuple):
        return M, theta_deg, score, nin
    if len(res) < 3:
        return M, theta_deg, score, nin
    M = res[0]
    tail = list(res[1:])
    # 提取 nin: 优先取 int；没有的话从尾部取近似整数
    nin_idx = None
    for i, v in enumerate(tail):
        if isinstance(v, (int, np.integer)):
            nin = int(v); nin_idx = i; break
    if nin_idx is None:
        # 没有明确的 int，就尝试把接近整数的最后一个当作 nin
        for i in reversed(range(len(tail))):
            v = tail[i]
            if isinstance(v, (float, np.floating)) and abs(v - round(v)) < 1e-6 and v >= 0:
                nin = int(round(v)); nin_idx = i; break
    if nin_idx is not None:
        tail.pop(nin_idx)
    # 现在 tail 应该有两个浮点: 角度 和 分数
    # 分数通常在 [0,1.5]，角度通常在 [-180, 180]
    cand = [float(x) for x in tail[:2]] + ([0.0] if len(tail)==1 else [])
    if len(cand) >= 2:
        a, b = cand[0], cand[1]
        # 试着判别谁是 score
        def is_score(x):
            return -0.05 <= x <= 1.5
        if is_score(a) and not is_score(b):
            score, theta_deg = a, b
        elif is_score(b) and not is_score(a):
            score, theta_deg = b, a
        else:
            # 都像/都不像分数，按常见顺序 (score, theta)
            score, theta_deg = a, b
    elif len(cand) == 1:
        # 只有一个值，优先当 score
        val = cand[0]
        if -0.05 <= val <= 1.5:
            score = val
        else:
            theta_deg = val
    return M, float(theta_deg), float(score), int(nin)

# Helper to extract actual ROI used from refine_alignment_multi_roi result, fallback to default
def _extract_roi_used(res, default_roi):
    """
    Try to get the actual ROI size used by refine_alignment_multi_roi from its return tuple.
    Backward compatible:
      - Old signatures: (M, score, nin) or (M, theta, score, nin) -> fall back to default_roi
      - New signature we added: (M, theta, score, nin, avg_roi[, ...]) -> use that avg_roi
    """
    roi_used = int(default_roi)
    try:
        if isinstance(res, tuple) and len(res) >= 5:
            candidate = res[4]
            if isinstance(candidate, (int, float, np.integer, np.floating)) and candidate > 0:
                roi_used = int(round(float(candidate)))
    except Exception:
        pass
    return roi_used

# ------------------ 调试图保存 ------------------
def save_debug_image(processed_img, target_center, reference_center,
                     shift_x, shift_y, confidence, method,
                     debug_dir: Path, filename, reference_filename):
    try:
        if processed_img is None:
            return
        if processed_img.ndim == 2:
            debug_image = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        else:
            debug_image = processed_img.copy()
        cv2.circle(debug_image, (int(target_center[0]), int(target_center[1])), 5, (0,0,255), -1)
        cv2.circle(debug_image, (int(reference_center[0]), int(reference_center[1])), 15, (0,255,255), 3)
        cv2.line(debug_image, (int(target_center[0]), int(target_center[1])),
                 (int(reference_center[0]), int(reference_center[1])), (0,255,255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        texts = [
            f"Method: {method[:35]}",
            f"Shift: ({shift_x:.1f}, {shift_y:.1f})",
            f"Confidence: {confidence:.3f}",
            f"Reference: {reference_filename}",
            f"Mode: Incremental Processing"
        ]
        for j, t in enumerate(texts):
            cv2.putText(debug_image, t, (10, 25 + j*25), font, 0.6, (255,255,255), 2)
        debug_path = debug_dir / f"debug_{filename}"
        imwrite_unicode(debug_path, debug_image)
    except Exception as e:
        print(f"调试图像生成失败: {e}")

# ------------------ 缩略图辅助 ------------------
def _detect_circle_on_thumb(img:np.ndarray, hough: Hough, max_side=1600, strong_denoise=False) ->  tuple[ Circle, float, float, str]:
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
    small = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img

    s_min = max(1, int(hough.min_radius * scale))
    s_max = max(s_min + 1, int(hough.max_radius * scale))

    t0 = time.time()
    circle_s, _, quality_s, method_s, _ = detect_circle_phd2_enhanced(small, s_min, s_max, p1, p2, strong_denoise=strong_denoise)
    dt = time.time() - t0

    if circle_s is None:
        raise Exception("缩略图圆检测失败")

    circle = Circle(
        x = float(circle_s[0] / scale),
        y = float(circle_s[1] / scale), 
        radius = float(circle_s[2] / scale)
    )
    return circle, scale, float(quality_s), f"{method_s}(thumb,{small.shape[1]}x{small.shape[0]}, {dt:.2f}s)"

# ------------------ 主流程 ------------------

def align_moon_images_incremental(input_dir: Path, output_dir: Path, hough: Hough,
                                 log_box=None, debug_mode=False, debug_image_basename="",
                                 completion_callback=None, progress_callback=None,
                                 reference_image_path=None, use_advanced_alignment=False,
                                 alignment_method='auto', strong_denoise=False):
    try:

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise Exception(f"无法创建输出文件夹: {output_dir}") from e

        debug_dir = output_dir / "debug"
        if debug_mode:
            try:
                debug_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise Exception(f"无法创建调试文件夹: {debug_dir}") from e

        try:
            image_files = sorted(itertools.chain.from_iterable(input_dir.glob(ext) for ext in SUPPORTED_EXTS))
        except Exception as e:
            raise Exception(f"读取输入文件夹失败: {e}")
        
        if not image_files:
            raise Exception(f"在 '{input_dir}' 中未找到支持的图片文件")

        min_rad, max_rad, param1, param2 = hough
        input_files_num = len(image_files)

        log("=" * 60, log_box)
        log(f"月食圆面对齐工具 V{VERSION} - 增量处理版", log_box)
        log(f"处理模式: 增量处理 (边检测边保存)", log_box)
        log(f"文件总数: {input_files_num}", log_box)
        log(f"多ROI精配准: {'启用' if use_advanced_alignment else '禁用'}", log_box)
        log("=" * 60, log_box)

        # 参考图像
        log("阶段 1/2: 确定参考图像...", log_box)
        reference_image = None
        reference_center = None
        reference_filename = None; best_quality = 0.0
        reference_radius = None

        # ---------- 用户指定参考图 ----------
        if reference_image_path and os.path.exists(reference_image_path):
            ref_filename = os.path.basename(reference_image_path)
            log(f"加载用户指定的参考图像: {ref_filename}", log_box)

            t_ref0 = time.time()
            ref_img = imread_unicode(reference_image_path, cv2.IMREAD_UNCHANGED)
            if ref_img is not None:
                H, W = ref_img.shape[:2]
                log(f"参考图尺寸: {W}x{H}", log_box)

                # 先在缩略图做，映射回原图
                try:
                    circle, scale, q, meth = _detect_circle_on_thumb(
                    ref_img, min_rad, max_rad, param1, param2, max_side=1600, strong_denoise=strong_denoise
                )
                
                    reference_image = ref_img.copy()
                    reference_center = (circle[0], circle[1])
                    reference_filename = ref_filename
                    best_quality = q
                    reference_radius = circle[2]
                    log(f"✓ 参考图像检测成功: 质量={q:.1f}, 方法={meth}, 半径≈{reference_radius:.1f}px", log_box)
                except Exception as e:
                    log("缩略图检测失败，回退到原图做一次圆检测（可能较慢）...", log_box)
                    t1 = time.time()
                    circle_full, _, qf, mf, _ = detect_circle_phd2_enhanced(
                        ref_img, min_rad, max_rad, param1, param2, strong_denoise=strong_denoise
                    )
                    dt1 = time.time() - t1
                    if circle_full is not None:
                        reference_image = ref_img.copy()
                        reference_circle = circle_full
                        reference_filename = ref_filename
                        best_quality = float(qf)
                        reference_radius = float(circle_full[2])
                        log(f"✓ 参考图像检测成功: 质量={best_quality:.1f}, 方法={mf}, 半径≈{reference_radius:.1f}px, 用时 {dt1:.2f}s", log_box)
                    else:
                        log("✗ 参考图像检测失败，将自动选择", log_box)
            else:
                log("✗ 参考图像读取失败，将自动选择", log_box)

        # ---------- 自动扫描前 N 张 ----------
        if reference_image is None:
            scan_count = min(10, input_files_num)
            log(f"自动选择参考图像 (扫描前{scan_count}张)...", log_box)
            for i, filename in enumerate(image_files[:scan_count]):
                if progress_callback:
                    progress_callback(int((i / scan_count) * 20), f"扫描参考图像: {filename}")
                input_path = safe_join(input_dir, filename)
                img0 = imread_unicode(input_path, cv2.IMREAD_UNCHANGED)
                if img0 is None:
                    continue
                try:
                    circle, scale, q, meth = _detect_circle_on_thumb(
                        img0, min_rad, max_rad, param1, param2, max_side=1600, strong_denoise=strong_denoise
                    )
                    if  q > best_quality:
                        reference_image = img0.copy()
                        reference_circle = circle
                        reference_filename = filename
                        best_quality = q
                        reference_radius = circle[2]
                        log(f"  候选参考图像: {filename}, 质量={q:.1f}, 方法={meth}", log_box)
                except Exception as e:
                    pass
                del img0
                force_garbage_collection()

        if reference_image is None:
            raise Exception("无法找到有效的参考图像，请检查图像质量和参数设置")

        log(f"🎯 最终参考图像: {reference_filename}, 质量评分={best_quality:.1f}", log_box)

        # 处理所有图像
        log(f"\n阶段 2/2: 增量处理所有图像...", log_box)
        success_count = 0; failed_files = []
        brightness_stats = {"bright": 0, "normal": 0, "dark": 0}
        method_stats = {}

        # 为速度统计
        t_all0 = time.time()

        # 以参考图圆作为先验，后续逐帧更新
        last_circle = None
        if reference_center is not None and reference_radius is not None:
            last_circle = (float(reference_center[0]), float(reference_center[1]), float(reference_radius))

        for i, filename in enumerate(image_files):
            if progress_callback:
                progress_callback(20 + int((i / input_files_num) * 80), f"处理: {filename}")
            try:
                input_path = safe_join(input_dir, filename)

                # 参考图：直接另存
                if filename == reference_filename:
                    output_path = safe_join(output_dir, f"aligned_{filename}")
                    if imwrite_unicode(output_path, reference_image):
                        success_count += 1
                        log(f"  🎯 {filename}: [参考图像] 已保存", log_box)
                        if debug_mode and filename == debug_image_basename:
                            save_debug_image(reference_image, reference_center, reference_center,
                                             0, 0, 1.0, "Reference Image",
                                             safe_join(output_dir, "debug"), filename, reference_filename)
                    else:
                        log(f"  ✗ {filename}: 保存失败", log_box); failed_files.append(filename)
                    continue

                # 读取目标
                t_read = time.time()
                target_image = imread_unicode(input_path, cv2.IMREAD_UNCHANGED)
                if target_image is None:
                    log(f"  ✗ {filename}: 读取失败", log_box); failed_files.append(filename); continue
                dt_read = time.time() - t_read

                # 圆检测
                t_det = time.time()
                circle, processed, quality, method, brightness = detect_circle_phd2_enhanced(
                    target_image, min_rad, max_rad, param1, param2,
                    strong_denoise=strong_denoise, prev_circle=last_circle
                )
                dt_det = time.time() - t_det

                if circle is None:
                    log(f"  ✗ {filename}: 圆检测失败(耗时 {dt_det:.2f}s)", log_box)
                    failed_files.append(filename); del target_image; continue

                brightness_stats[brightness] += 1
                method_stats[method] = method_stats.get(method, 0) + 1

                target_center = (circle[0], circle[1])

                # 初始：圆心平移到参考
                shift_x = reference_center[0] - target_center[0]
                shift_y = reference_center[1] - target_center[1]
                confidence = max(0.30, min(0.98, quality / 100.0))
                align_method = "Circle Center"
                theta_deg = 0.0

                rows, cols = target_image.shape[:2]
                M = np.float32([[1,0,shift_x],[0,1,shift_y]])
                aligned = cv2.warpAffine(target_image, M, (cols, rows),
                                         flags=cv2.INTER_LANCZOS4,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                # 多 ROI 精配准（仅平移，无旋转）
                try:
                    if reference_radius is not None and use_advanced_alignment:
                        ref_gray = reference_image if reference_image.ndim==2 else cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
                        tgt_gray2 = aligned if aligned.ndim==2 else cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

                        roi_size = max(64, min(160, int(reference_radius*0.18)))
                        max_refine_delta_px = 6.0
                        t_refine = time.time()
                        res = refine_alignment_multi_roi(
                            ref_gray, tgt_gray2,
                            float(reference_center[0]), float(reference_center[1]),
                            float(reference_radius),
                            n_rois=16, roi_size=roi_size, search=12,
                            use_phasecorr=True, use_ecc=False,
                            base_shift=(float(shift_x), float(shift_y)),
                            max_refine_delta_px=max_refine_delta_px
                        )
                        dt_refine = time.time() - t_refine
                        M2, theta_deg, score, nin = _unpack_refine_result(res)
                        roi_used = _extract_roi_used(res, roi_size)
                        log(f"    [Refine] score={score:.3f}, inliers={nin}, roi_init≈{roi_used}, t={dt_refine:.2f}s", log_box)
                        residual = None
                        if M2 is not None:
                            tx = float(M2[0,2])
                            ty = float(M2[1,2])
                            residual = (tx**2 + ty**2) ** 0.5
                            log(f"    [Refine] 残差=Δ{residual:.2f}px", log_box)
                            if residual > max_refine_delta_px:
                                M2 = None
                                log(f"    [Refine] 残差过大(Δ={residual:.2f}px > {max_refine_delta_px:.1f}px)，放弃精配准并保持霍夫平移", log_box)
                        if M2 is not None:
                            aligned = cv2.warpAffine(
                                aligned, M2, (cols, rows),
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0
                            )
                            confidence = max(confidence, float(score))
                            align_method = f"Multi-ROI refine (仅平移, inliers={nin}, roi_init≈{roi_used}, Δ={residual:.2f}px, gate≤{max_refine_delta_px:.0f}px, {dt_refine:.2f}s)"
                        else:
                            log("    [Refine] 无有效解，回退 Masked PhaseCorr", log_box)
                            # 遮罩相位相关微调（仅平移）
                            t_pc = time.time()
                            dx2, dy2 = masked_phase_corr(
                                ref_gray, tgt_gray2,
                                float(reference_center[0]), float(reference_center[1]),
                                float(reference_radius)
                            )
                            dt_pc = time.time() - t_pc
                            if abs(dx2)>1e-3 or abs(dy2)>1e-3:
                                M2 = np.float32([[1,0,dx2],[0,1,dy2]])
                                aligned = cv2.warpAffine(aligned, M2, (cols, rows),
                                                         flags=cv2.INTER_LANCZOS4,
                                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                align_method = f"Masked PhaseCorr ({dt_pc:.2f}s)"
                    elif reference_radius is not None:
                        # 未启用高级：遮罩相位相关微调
                        ref_gray = reference_image if reference_image.ndim==2 else cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
                        tgt_gray2 = aligned if aligned.ndim==2 else cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                        dx2, dy2 = masked_phase_corr(
                            ref_gray, tgt_gray2,
                            float(reference_center[0]), float(reference_center[1]),
                            float(reference_radius)
                        )
                        if abs(dx2)>1e-3 or abs(dy2)>1e-3:
                            M2 = np.float32([[1,0,dx2],[0,1,dy2]])
                            aligned = cv2.warpAffine(aligned, M2, (cols, rows),
                                                     flags=cv2.INTER_LANCZOS4,
                                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                            align_method = "Masked PhaseCorr"
                except Exception as e:
                    log(f"    [Refine异常] {filename}: {e}", log_box)

                # 保存
                out_path = safe_join(output_dir, f"aligned_{filename}")
                if imwrite_unicode(out_path, aligned):
                    success_count += 1
                    # 更新上一帧先验
                    try:
                        last_circle = (float(circle[0]), float(circle[1]), float(circle[2]))
                    except Exception:
                        pass
                    log(f"  ✓ {filename}: 偏移=({shift_x:.1f},{shift_y:.1f}), "
                        f"质量={quality:.1f}, 置信度={confidence:.3f}, 圆检耗时={dt_det:.2f}s, 读取={dt_read:.2f}s | {align_method}", log_box)

                    if debug_mode and filename == debug_image_basename and processed is not None:
                        save_debug_image(processed, target_center, reference_center,
                                         shift_x, shift_y, confidence, align_method,
                                         debug_dir, filename, reference_filename)
                else:
                    log(f"  ✗ {filename}: 变换成功但保存失败", log_box)
                    failed_files.append(filename)

                del target_image, aligned
                if 'processed' in locals(): del processed
                force_garbage_collection()

            except Exception as e:
                log(f"  ✗ {filename}: 处理异常 - {e}", log_box)
                failed_files.append(filename)
                for v in ['target_image','aligned','processed']:
                    if v in locals(): del locals()[v]
                force_garbage_collection()

        if progress_callback: progress_callback(100, "处理完成")
        del reference_image; force_garbage_collection()

        log("=" * 60, log_box)
        log(f"增量对齐完成! 成功对齐 {success_count}/{input_files_num} 张图像", log_box)
        log(f"使用参考图像: {reference_filename}", log_box)
        log(f"对齐算法: {'多ROI精配准（仅平移）' if use_advanced_alignment else 'PHD2圆心算法'}", log_box)
        if failed_files:
            head = ', '.join(failed_files[:5]) + ("..." if len(failed_files)>5 else "")
            log(f"失败文件({len(failed_files)}): {head}", log_box)
        if method_stats:
            log("圆检测方法统计: " + ', '.join([f"{k}={v}" for k,v in method_stats.items()]), log_box)
        log(f"当前内存使用: {get_memory_usage_mb():.1f} MB", log_box)
        if completion_callback:
            completion_callback(True, f"增量处理完成！成功对齐 {success_count}/{input_files_num} 张图像")

    except Exception as e:
        import traceback
        err = f"增量处理过程中发生错误: {e}\n{traceback.format_exc()}"
        log(err, log_box)
        if completion_callback:
            completion_callback(False, err)
    finally:
        force_garbage_collection()
