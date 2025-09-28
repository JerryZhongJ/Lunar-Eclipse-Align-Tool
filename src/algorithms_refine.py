from dataclasses import dataclass
import logging
import cv2
import numpy as np
import math
import time
from utils import ROI, Circle, Position, Vector, VectorArray
from numpy.typing import NDArray

# ---------------- 工具函数 ----------------


def _soft_disk_mask(h, w, circle: Circle, inner=0.0, outer=0.98) -> NDArray:
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - circle.x) ** 2 + (Y - circle.y) ** 2)
    m = np.zeros((h, w), np.float32)
    r_in = circle.radius * max(0.0, inner)
    r_out = circle.radius * min(1.0, outer)
    core = dist <= (r_out * 0.90)
    m[core] = 1.0
    band = (dist > (r_out * 0.90)) & (dist <= r_out)
    if np.any(band):
        t = (dist[band] - r_out * 0.90) / (r_out * 0.10 + 1e-6)
        m[band] = 0.5 * (1 + np.cos(np.pi * t))
    if r_in > 1:
        m[dist < r_in] = 0
    return m


def _clahe_and_bandpass(gray: np.ndarray):
    g = gray.copy()
    # Ensure 8-bit input
    if g.dtype != np.uint8:
        cv2.normalize(g.astype(np.float32), g, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Local contrast to suppress global illumination, emphasize small-scale details
    cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)

    # High-pass only (remove mid/low frequencies): unsharp mask
    hp = cv2.subtract(
        cla, cv2.GaussianBlur(cla, (0, 0), 6.0)
    )  # sigma ~6 -> keep only high freq

    # Edge emphasis via Laplacian (pure high-frequency operator)
    lap = cv2.Laplacian(cla, cv2.CV_32F, ksize=3)

    # Combine and normalize
    combo = 0.5 * np.abs(hp.astype(np.float32)) + 0.5 * np.abs(lap)
    cv2.normalize(combo, combo, 0, 255, cv2.NORM_MINMAX)
    return combo.astype(np.uint8)


def _match_roi_zncc_local(
    ref_patch: NDArray,
    tgt_img: NDArray,
    pos: Position[int],
    search: int = 12,
    mask_patch: NDArray | None = None,
) -> tuple[Vector[int], float]:
    """Match ref_patch around (x,y) in tgt_img within +/-search window.
    Uses TM_CCORR_NORMED with optional template mask (same size as ref_patch).
    Returns (dx, dy, score).
    """
    h, w = ref_patch.shape
    H, W = tgt_img.shape

    # target search window centered at the ref patch center
    c: Position[int] = pos + Vector(w // 2, h // 2)
    pos0: Position[int] = Position(
        max(0, c.x - w // 2 - search), max(0, c.y - h // 2 - search)
    )
    pos1: Position[int] = Position(
        min(W, pos0.x + w + 2 * search), min(H, pos0.y + h + 2 * search)
    )

    crop = tgt_img[pos0.y : pos1.y, pos0.x : pos1.x]
    zero_vector = Vector(0, 0)
    if crop.shape[0] < h or crop.shape[1] < w:
        return zero_vector, -1.0

    tpl = ref_patch.astype(np.float32)
    win = crop.astype(np.float32)

    # Variance guard: if the (masked) template has almost no texture, skip
    if mask_patch is not None and mask_patch.shape == ref_patch.shape:
        m = mask_patch.astype(np.float32) / 255.0
        area = m.sum()
        if area < 16:  # too small effective area
            return zero_vector, -1.0
        tpl_eff = tpl[m > 0.5]
        if tpl_eff.size == 0 or np.std(tpl_eff) < 1e-3:
            return zero_vector, -1.0
        method = cv2.TM_CCORR_NORMED
        res = cv2.matchTemplate(win, tpl, method, mask=mask_patch.astype(np.uint8))
    else:
        if np.std(tpl) < 1e-3:
            return zero_vector, -1.0
        method = cv2.TM_CCORR_NORMED
        res = cv2.matchTemplate(win, tpl, method)

    # Handle NaN/Inf from OpenCV edge cases
    if not np.isfinite(res).all():
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)

    minv, maxv, minloc, maxloc = cv2.minMaxLoc(res)
    maxloc = Vector(maxloc[0], maxloc[1])
    d = pos0 + maxloc + Vector(w // 2, h // 2) - c

    return d, maxv


from utils import Circle


def _select_rois(
    energy_map: NDArray,
    disk_mask: NDArray[np.bool],
    r: float,
    k: int = 16,
    box: int = 128,
    border: int = 10,
    avoid_edge_ratio: float = 0.06,
    ref_img=None,
    brightness_range=(30, 220),
) -> list[ROI]:
    """Pick top-k ROI boxes inside the lunar disk by local energy, spaced apart.
    Supports adaptive ROI size based on local brightness if ref_img is provided.
    """
    h, w = energy_map.shape

    rois: list[ROI] = []
    # keep away from limb/edge a little bit
    margin = max(border, int(r * avoid_edge_ratio))
    step = max(24, box // 2)  # stride to reduce overlap
    integ = cv2.integral(energy_map.astype(np.float32))

    def sum_rect(x0, y0, bw, bh):
        # integral image sum over rectangle [x0,x0+bw) x [y0,y0+bh)
        return float(
            integ[y0 + bh, x0 + bw]
            - integ[y0, x0 + bw]
            - integ[y0 + bh, x0]
            + integ[y0, x0]
        )

    for y in range(margin, h - margin - box, step):
        for x in range(margin, w - margin - box, step):
            submask = disk_mask[y : y + box, x : x + box]
            if submask.mean() < 0.6:
                continue
            # Determine adaptive box size based on local brightness if ref_img provided
            local_box = box
            if ref_img is not None:
                roi_block = ref_img[y : y + box, x : x + box]
                bmin, bmax = brightness_range
                frac_in = np.mean((roi_block >= bmin) & (roi_block <= bmax))
                if frac_in < 0.4:
                    continue
                # Compute mean brightness in the ROI region
                roi_brightness = np.mean(roi_block)
                # Map brightness to ROI size: brighter -> larger ROI, darker -> smaller ROI
                min_box = max(64, int(box * 0.5))
                max_box = box
                bmin, bmax = brightness_range
                # Clamp brightness
                b = np.clip(roi_brightness, bmin, bmax)
                scale = (b - bmin) / (bmax - bmin)
                local_box = int(min_box + scale * (max_box - min_box))
                # Adjust local_box to be multiple of 8 for consistency
                local_box = (local_box // 8) * 8
                if local_box < 64:
                    local_box = 64
                if y + local_box > h - margin or x + local_box > w - margin:
                    continue
                submask = disk_mask[y : y + local_box, x : x + local_box]
                if submask.mean() < 0.6:
                    continue
                score = sum_rect(x, y, local_box, local_box) / (
                    local_box * local_box + 1e-6
                )
                rois.append(ROI(x, y, local_box, local_box, score))
            else:
                score = sum_rect(x, y, box, box) / (box * box + 1e-6)
                rois.append(ROI(x, y, box, box, score))

    rois.sort(key=lambda t: t.score, reverse=True)

    pickeds: list[ROI] = []
    for roi in rois:
        too_close = False
        for picked in pickeds:
            min_sep_x = min(roi.w, picked.w) // 2
            min_sep_y = min(roi.h, picked.h) // 2
            if abs(roi.x - picked.x) < min_sep_x and abs(roi.y - picked.y) < min_sep_y:
                too_close = True
                break
        if not too_close:
            pickeds.append(roi)
        if len(pickeds) >= k:
            break
    return pickeds


def refine_alignment_multi_roi(
    ref_gray: NDArray,
    tgt_gray: NDArray,
    circle: Circle,
    n_rois=16,
    roi_size=128,
    search: int = 12,
    base_shift: Vector | None = None,
    max_refine_delta_px=6.0,
    min_inliers=6,
    min_mean_zncc=0.55,
    use_phasecorr=True,
    use_ecc=False,
    time_budget_sec=1.2,
    debug_cb=None,
) -> tuple[NDArray[np.float32], float, int, float]:
    """
    输入：已做圆心粗配准的 ref_gray / tgt_gray，以及参考圆心(cx,cy)与半径 r
    输出：(M2x3, score, n_inliers, theta_deg)
    特性：
      • 仅估计 旋转+平移（无缩放），绕 (cx,cy) 旋转
      • ROI 只在局部窗口内做匹配，避免全图卷积导致卡顿
      • 内置时间预算，个别困难帧自动提前结束并回退上层策略
      • 过滤过暗/低纹理 ROI，提升稳健性
    """
    H, W = ref_gray.shape
    logging.debug(
        f"[Refine] HxW={H}x{W}, r≈{circle.radius:.1f}, n_rois={n_rois}, roi_init={roi_size}, search={search}"
    )
    logging.debug(f"[Refine] 启用自适应ROI与亮度门控: brightness∈[30,220]")

    # 软盘遮罩，仅保留月盘内纹理
    mask = _soft_disk_mask(H, W, circle, inner=0.0, outer=0.97)

    # 归一化转为8-bit，用于能量图和ROI选择
    # ref_gray = ref_gray.copy()
    if ref_gray.dtype != np.uint8:
        cv2.normalize(ref_gray.astype(np.float32), ref_gray, 0, 255, cv2.NORM_MINMAX)
        ref_gray = ref_gray.astype(np.uint8)
    # tgt_gray = tgt_gray.copy()
    if tgt_gray.dtype != np.uint8:
        cv2.normalize(tgt_gray.astype(np.float32), tgt_gray, 0, 255, cv2.NORM_MINMAX)
        tgt_gray = tgt_gray.astype(np.uint8)

    # 梯度/DoG 预处理（抗亮度变化），用于匹配阶段
    refF = _clahe_and_bandpass(ref_gray)
    tgtF = _clahe_and_bandpass(tgt_gray)

    # 只保留盘内
    refF = (refF.astype(np.float32) * mask).astype(np.uint8)
    tgtF = (tgtF.astype(np.float32) * mask).astype(np.uint8)

    # 能量图直接用高频响应（仅保留高频，避免中频干扰）
    energy = cv2.GaussianBlur(refF, (0, 0), 1.2)

    # 自适应/裁剪 ROI 数量与大小，避免过多卷积
    n_rois = int(np.clip(n_rois if n_rois else (circle.radius / 70), 8, 24))
    roi_size = int(np.clip(roi_size if roi_size else (circle.radius * 0.12), 64, 128))
    search = int(np.clip(search if search else (circle.radius * 0.05), 6, 18))

    rois: list[ROI] = _select_rois(
        energy, mask, circle.radius, k=n_rois, box=roi_size, ref_img=ref_gray
    )
    logging.debug(f"[Refine] ROI候选数={len(rois)}")

    sizes = [roi.w for roi in rois]
    if sizes:
        logging.debug(
            f"[Refine] 自适应ROI统计: 平均={np.mean(sizes):.0f}, 最小={np.min(sizes)}, 最大={np.max(sizes)}"
        )
    if not rois:
        raise Exception()

    # 收集 ROI 局部位移向量（dx_i, dy_i）及其中心（xi, yi）
    centers: list[Position[float]] = []
    zncc_list = []

    d_list: list[Vector[float]] = []
    weights: list[float] = []
    t_start = time.perf_counter()

    for roi in rois:
        # 时间预算：超时则提前结束，防止个别难帧拖慢
        if time.perf_counter() - t_start > time_budget_sec:
            break

        ref_patch = refF[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]

        # —— 过滤过暗/低纹理 ROI（避免落在阴影/背景）——
        m = float(ref_patch.mean())
        s = float(ref_patch.std())
        if m < 8.0 or s < 6.0:  # 可按画面调节
            logging.debug(f"[Refine] 丢弃ROI: 过暗/低纹理 mean={m:.1f}, std={s:.1f}")
            continue

        bmin, bmax = 30, 220
        ref_block8 = ref_gray[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
        mask_patch = ((ref_block8 >= bmin) & (ref_block8 <= bmax)).astype(
            np.uint8
        ) * 255
        if mask_patch.size > 0:
            mask_patch = cv2.erode(mask_patch, np.ones((3, 3), np.uint8), iterations=1)

        vd, zncc = _match_roi_zncc_local(
            ref_patch, tgtF, roi, search=search, mask_patch=mask_patch
        )
        vd = vd.as_type(float)
        if zncc < 0.28:
            logging.debug(f"[Refine] 丢弃ROI: ZNCC过低 zncc={zncc:.2f}")
            continue

        # 丢弃命中搜索边界的解，通常是含糊/假峰
        if (abs(vd.x) >= (search - 0.5)) or (abs(vd.y) >= (search - 0.5)):
            logging.debug(
                f"[Refine] 丢弃ROI: 命中搜索边界 dx={vd.x:.2f}, dy={vd.y:.2f}"
            )

            continue

        # 可选：相位相关做亚像素微调（仅在纹理足够时启用）
        if use_phasecorr:
            h, w = ref_patch.shape
            cp: Position[int] = ROI(roi.x, roi.y, w, h).center.as_type(int)
            p: Position[int] = cp - Vector(w // 2 + search, h // 2 + search)
            e: Position[int] = Position(
                min(tgtF.shape[1], p.x + w), min(tgtF.shape[0], p.y + h)
            )

            tgt_patch = tgtF[p.y : e.y, p.x : e.x]
            if tgt_patch.shape == ref_patch.shape:
                rp = ref_patch.astype(np.float32)
                tp = tgt_patch.astype(np.float32)
                if mask_patch is not None and mask_patch.shape == ref_patch.shape:
                    m = mask_patch.astype(np.float32) / 255.0
                    rp = rp * m
                    tp = tp * m
                    # 纹理/有效面积检查
                    eff = rp[m > 0.5]
                    if eff.size == 0 or np.std(eff) < 1e-3:
                        tgt_patch = None
                if tgt_patch is not None:
                    (dx2, dy2), resp = cv2.phaseCorrelate(rp, tp)
                    if (
                        (zncc >= 0.60)
                        and (resp is not None)
                        and np.isfinite(resp)
                        and (resp > 0.20)
                        and (abs(dx2) < 2.5)
                        and (abs(dy2) < 2.5)
                    ):
                        vd = vd + Vector(dx2, dy2)
                        zncc = max(zncc, float(resp))

        centers.append(roi.center)
        d_list.append(vd)

        # 结合对比度作为权重，弱纹理权重低
        weights.append(
            float(max(1e-3, zncc) * (0.5 + 0.5 * np.clip(s / 20.0, 0.0, 1.0)))
        )
        zncc_list.append(float(zncc))
        logging.debug(
            f"[Refine] 采纳ROI: zncc={zncc:.2f}, dx={vd.x:.2f}, dy={vd.y:.2f}"
        )

    n = len(centers)
    logging.debug(f"[Refine] 参与拟合的ROI数={n}")

    if n < 3:
        # Not enough ROIs; if baseline (Hough) is provided, fall back to it
        if base_shift is not None:
            th = base_shift
            M_h = np.array([[1.0, 0.0, th.x], [0.0, 1.0, th.y]], dtype=np.float32)
            logging.debug(f"[Refine] ROI不足，回退霍夫: shift=({th.x:.2f},{th.y:.2f})")
            return M_h, 0.0, 0, 0.0
        raise Exception()

    w_arr = np.asarray(weights, dtype=np.float64)
    d_arr = VectorArray(np.array(d_list), safe=False)
    zncc_arr = (
        np.asarray(zncc_list, dtype=np.float64)
        if len(zncc_list)
        else np.array([], dtype=np.float64)
    )
    mean_zncc = float(zncc_arr.mean()) if zncc_arr.size else 0.0

    # —— 仅估计平移（无旋转），使用Tukey双权IRLS（单帧内稳健，无跨帧约束）——

    med = np.median(d_arr._arr, axis=0)
    med = Vector.from_ndarray(med, float)
    resid = (d_arr - med).norms()

    mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
    sigma = 1.4826 * mad + 1e-6
    c = 4.685 * sigma  # Tukey截止

    # 初始权（来自匹配质量）
    base_w = np.clip(w_arr, 1e-6, None)

    # 一次IRLS即可（经验上已足够稳健）
    r = (d_arr - med).norms()
    tukey = np.zeros_like(r)
    m = r < c
    rr = r[m] / (c + 1e-6)
    tukey[m] = (1 - rr**2) ** 2

    ww = base_w * tukey
    if ww.sum() < 1e-6 or (tukey > 0).sum() < 3:
        if base_shift is not None:
            th = base_shift
            M_h = np.array([[1.0, 0.0, th.x], [0.0, 1.0, th.y]], dtype=np.float32)
            logging.debug(f"[Refine] IRLS失败，回退霍夫: shift=({th.x:.2f},{th.y:.2f})")
            return M_h, 0.0, 0, 0.0
        raise Exception()

    t = Vector.from_ndarray(np.average(d_arr._arr, axis=0, weights=ww), float)
    cnt = int((tukey > 0).sum())
    score = float((cnt / float(n)) * (ww.max() / (base_w.max() + 1e-6)))

    # --- Quality gate & baseline fallback (no inter-frame constraint) ---
    if base_shift is not None:
        dh = base_shift
        delta = (t - dh).norm()
        bad = False
        reasons = []
        if cnt < int(min_inliers):
            bad = True
            reasons.append(f"inliers={cnt}<{int(min_inliers)}")
        if mean_zncc < float(min_mean_zncc):
            bad = True
            reasons.append(f"mean_zncc={mean_zncc:.2f}<{float(min_mean_zncc):.2f}")
        if delta > float(max_refine_delta_px):
            bad = True
            reasons.append(
                f"|refine-hough|={delta:.2f}px>{float(max_refine_delta_px):.2f}"
            )
        if bad:
            # Fall back to Hough baseline shift
            t = dh
            logging.debug(
                "[Refine] 触发门控，回退霍夫: "
                + "; ".join(reasons)
                + f" -> ({t.x:.2f},{t.y:.2f})"
            )

    # 最终 2x3 仿射矩阵（仅平移）
    M = np.array([[1.0, 0.0, t.x], [0.0, 1.0, t.y]], dtype=np.float32)

    theta_deg = 0.0
    logging.debug(
        f"[Refine] 内点={cnt}/{n}, 平移=({t.x:.2f},{t.y:.2f}), meanZNCC={mean_zncc:.2f}, score={score:.3f}"
    )
    return M, score, int(cnt), theta_deg
