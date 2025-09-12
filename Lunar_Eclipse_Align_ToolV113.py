#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import sys
import platform
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import gc
from PIL import Image, ImageTk

import math
import random

try:
    from scipy.fft import fft2, ifft2
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy未安装，将禁用相位相关算法")

try:
    from ttkthemes import ThemedTk
except ImportError:
    ThemedTk = None

# 月食圆面对齐工具 V1.1.3 - 集成版
# 结合PHD2增强对齐、内存管理、跨平台兼容性
# 以及IMPPG高级对齐算法
# 基于原版本整合优化
# 本代码由 @正七价的氟离子 原始创建，ChatGPT、Manus AI、Claude优化与注释

# ----------------- 系统兼容性设置 -----------------
SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_MACOS = SYSTEM == "Darwin"
IS_LINUX = SYSTEM == "Linux"

# 根据系统设置默认字体
if IS_WINDOWS:
    DEFAULT_FONT = ("Microsoft YaHei", 9)
    UI_FONT = ("Microsoft YaHei", 9)
elif IS_MACOS:
    DEFAULT_FONT = ("SF Pro Display", 13)
    UI_FONT = ("SF Pro Display", 13)
else:
    DEFAULT_FONT = ("DejaVu Sans", 9)
    UI_FONT = ("DejaVu Sans", 9)

# ----------------- 全局默认值 -----------------
VERSION = "1.1.3"
DEFAULT_DEBUG_MODE = False
DEFAULT_DEBUG_IMAGE_PATH = ""
SUPPORTED_EXTS = {'.tif', '.tiff', '.bmp', '.png', '.jpg', '.jpeg'}

# 内存管理设置
MAX_IMAGES_IN_MEMORY = 10
MEMORY_THRESHOLD_MB = 500

# ----------------- 内存管理工具 -----------------
def get_memory_usage_mb():
    """获取当前进程内存使用量（MB）"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0

def force_garbage_collection():
    """强制垃圾回收"""
    gc.collect()

class MemoryManager:
    """内存管理器"""
    def __init__(self, threshold_mb=MEMORY_THRESHOLD_MB):
        self.threshold_mb = threshold_mb
        self.image_cache = {}
        self.access_order = []
    
    def should_clear_memory(self):
        """检查是否需要清理内存"""
        current_mb = get_memory_usage_mb()
        return current_mb > self.threshold_mb or len(self.image_cache) > MAX_IMAGES_IN_MEMORY
    
    def clear_old_images(self, keep_count=5):
        """清理旧的图像缓存"""
        if len(self.access_order) > keep_count:
            to_remove = self.access_order[:-keep_count]
            for key in to_remove:
                if key in self.image_cache:
                    del self.image_cache[key]
                self.access_order.remove(key)
        force_garbage_collection()

# ----------------- 路径处理工具 -----------------
def normalize_path(path):
    """标准化路径，确保跨平台兼容性"""
    if not path:
        return path
    path = path.replace('\\', os.sep).replace('/', os.sep)
    return os.path.normpath(path)

def ensure_dir_exists(dir_path):
    """确保目录存在，支持中文路径"""
    try:
        dir_path = normalize_path(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"创建目录失败: {e}")
        return False

def safe_join(*paths):
    """安全的路径连接"""
    return normalize_path(os.path.join(*paths))

# ----------------- 核心图像处理函数 -----------------

def imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    """支持中文路径和跨平台的安全读取"""
    try:
        path = normalize_path(path)
        
        if not IS_WINDOWS or path.isascii():
            img = cv2.imread(path, flags)
            if img is not None:
                return img
        
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, flags)
            if img is not None:
                return img
        except Exception:
            pass
        
        return cv2.imread(path, flags)
    except Exception as e:
        print(f"图像读取失败 {path}: {e}")
        return None

def imwrite_unicode(path, image):
    """跨平台、中文路径兼容的图像保存"""
    try:
        path = normalize_path(path)
        
        parent_dir = os.path.dirname(path)
        if not ensure_dir_exists(parent_dir):
            return False

        ext = os.path.splitext(path)[1].lower()
        if not ext:
            ext = ".tif"
            path = path + ext

        if not IS_WINDOWS or path.isascii():
            if ext in (".tif", ".tiff"):
                params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
                return cv2.imwrite(path, image, params)
            else:
                return cv2.imwrite(path, image)
        else:
            if ext in (".tif", ".tiff"):
                params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
                ok, buf = cv2.imencode(".tif", image, params)
            else:
                ok, buf = cv2.imencode(ext, image)
            
            if ok:
                buf.tofile(path)
                return True
            return False
            
    except Exception as e:
        print(f"图像保存失败 {path}: {e}")
        return False

def to_display_rgb(img):
    """将任意图像安全地转换为8位RGB用于GUI显示"""
    if img is None:
        return None
    
    try:
        img_float = img.astype(np.float32)
        img_u8 = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        if img_u8.ndim == 2:
            return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
        elif img_u8.shape[2] == 4:
            return cv2.cvtColor(img_u8, cv2.COLOR_BGRA2RGB)
        elif img_u8.shape[2] == 3:
            return cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB)
        else:
            return cv2.cvtColor(img_u8[:,:,:3], cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"图像转换失败: {e}")
        return None

# ----------------- IMPPG高级对齐算法 -----------------

def simple_phase_correlation(img1, img2):
    """简化版相位相关算法 - 重点保证稳定性"""
    if not SCIPY_AVAILABLE:
        return 0, 0, 0
        
    try:
        if img1.shape != img2.shape:
            return 0, 0, 0
            
        h, w = img1.shape[:2]
        
        f1 = img1.astype(np.float32)
        f2 = img2.astype(np.float32)
        
        F1 = np.fft.fft2(f1)
        F2 = np.fft.fft2(f2)
        
        cross_power_spectrum = F1 * np.conj(F2)
        magnitude = np.abs(cross_power_spectrum)
        magnitude = np.where(magnitude > 1e-10, magnitude, 1e-10)
        cross_power_spectrum = cross_power_spectrum / magnitude
        
        correlation = np.real(np.fft.ifft2(cross_power_spectrum))
        peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        shift_x = peak_x if peak_x < w/2 else peak_x - w
        shift_y = peak_y if peak_y < h/2 else peak_y - h
        
        max_shift = min(w, h) * 0.4
        if abs(shift_x) > max_shift or abs(shift_y) > max_shift:
            return 0, 0, 0.1
        
        max_corr = np.max(correlation)
        mean_corr = np.mean(correlation)
        confidence = min(1.0, (max_corr - mean_corr) / (max_corr + 1e-10))
        
        return float(shift_x), float(shift_y), max(0.2, float(confidence))
        
    except Exception as e:
        print(f"相位相关失败: {e}")
        return 0, 0, 0

def template_matching_alignment(img1, img2):
    """模板匹配算法 - 作为简单可靠的备选方案"""
    try:
        h, w = img1.shape[:2]
        
        template_size = min(h, w) // 4
        center_x, center_y = w // 2, h // 2
        
        template = img1[center_y - template_size//2:center_y + template_size//2,
                       center_x - template_size//2:center_x + template_size//2]
        
        if template.size == 0:
            return 0, 0, 0
        
        result = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        shift_x = max_loc[0] + template_size//2 - center_x
        shift_y = max_loc[1] + template_size//2 - center_y
        
        max_shift = min(w, h) * 0.3
        if abs(shift_x) > max_shift or abs(shift_y) > max_shift:
            return 0, 0, 0
        
        confidence = max(0.1, float(max_val))
        
        return float(shift_x), float(shift_y), confidence
        
    except Exception as e:
        print(f"模板匹配失败: {e}")
        return 0, 0, 0

def feature_matching_alignment(img1, img2):
    """简化的特征匹配算法"""
    try:
        orb = cv2.ORB_create(nfeatures=1000)
        
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            return 0, 0, 0
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 5:
            return 0, 0, 0
        
        src_pts = np.array([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.array([kp2[m.trainIdx].pt for m in matches])
        
        shifts_x = src_pts[:, 0] - dst_pts[:, 0]
        shifts_y = src_pts[:, 1] - dst_pts[:, 1]
        
        shift_x = np.median(shifts_x)
        shift_y = np.median(shifts_y)
        
        consistency = np.mean(np.abs(shifts_x - shift_x) < 10) * np.mean(np.abs(shifts_y - shift_y) < 10)
        confidence = min(1.0, len(matches) / 50.0 * consistency)
        
        return float(shift_x), float(shift_y), float(confidence)
        
    except Exception as e:
        print(f"特征匹配失败: {e}")
        return 0, 0, 0

def robust_centroid_alignment(img1, img2):
    """鲁棒的重心对齐算法"""
    try:
        def compute_centroid(img):
            mean_val = np.mean(img)
            std_val = np.std(img)
            
            thresholds = [
                mean_val + 0.5 * std_val,
                mean_val + std_val,
                np.percentile(img, 75)
            ]
            
            best_centroid = None
            best_area = 0
            
            for thresh in thresholds:
                mask = img > thresh
                area = np.sum(mask)
                
                if area > 100 and (best_area == 0 or 1000 < area < best_area * 3):
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) > 0:
                        weights = img[mask].astype(np.float64)
                        cx = np.average(x_indices, weights=weights)
                        cy = np.average(y_indices, weights=weights)
                        
                        best_centroid = (cx, cy)
                        best_area = area
            
            return best_centroid, best_area
        
        centroid1, area1 = compute_centroid(img1)
        centroid2, area2 = compute_centroid(img2)
        
        if centroid1 is None or centroid2 is None:
            return 0, 0, 0
        
        shift_x = centroid1[0] - centroid2[0]
        shift_y = centroid1[1] - centroid2[1]
        
        area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        confidence = area_ratio * 0.7
        
        return shift_x, shift_y, confidence
        
    except Exception as e:
        print(f"重心对齐失败: {e}")
        return 0, 0, 0

def multi_method_alignment(ref_image, target_image, method='auto', log_callback=None):
    """多方法对齐，确保有效的回退机制"""
    def log_debug(msg):
        if log_callback:
            log_callback(f"    {msg}")
    
    # 预处理
    if len(ref_image.shape) > 2:
        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_image.copy()
    
    if len(target_image.shape) > 2:
        target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    else:
        target_gray = target_image.copy()

    if ref_gray.dtype != np.uint8:
        ref_gray = cv2.normalize(ref_gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if target_gray.dtype != np.uint8:
        target_gray = cv2.normalize(target_gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if ref_gray.shape != target_gray.shape:
        target_gray = cv2.resize(target_gray, (ref_gray.shape[1], ref_gray.shape[0]))

    results = []
    
    # 根据方法选择执行对齐算法
    if method in ['auto', 'phase_corr']:
        log_debug("尝试相位相关算法...")
        shift_x, shift_y, conf = simple_phase_correlation(ref_gray, target_gray)
        if conf > 0.1:
            results.append(('Phase Correlation', shift_x, shift_y, conf))
            log_debug(f"相位相关: 偏移({shift_x:.1f}, {shift_y:.1f}), 置信度={conf:.3f}")
        else:
            log_debug("相位相关算法失败")

    if method in ['auto', 'template']:
        log_debug("尝试模板匹配算法...")
        shift_x, shift_y, conf = template_matching_alignment(ref_gray, target_gray)
        if conf > 0.1:
            results.append(('Template Matching', shift_x, shift_y, conf))
            log_debug(f"模板匹配: 偏移({shift_x:.1f}, {shift_y:.1f}), 置信度={conf:.3f}")
        else:
            log_debug("模板匹配算法失败")

    if method in ['auto', 'feature']:
        log_debug("尝试特征匹配算法...")
        shift_x, shift_y, conf = feature_matching_alignment(ref_gray, target_gray)
        if conf > 0.15:
            results.append(('Feature Matching', shift_x, shift_y, conf))
            log_debug(f"特征匹配: 偏移({shift_x:.1f}, {shift_y:.1f}), 置信度={conf:.3f}")
        else:
            log_debug("特征匹配算法失败")

    if method in ['auto', 'centroid']:
        log_debug("尝试重心对齐算法...")
        shift_x, shift_y, conf = robust_centroid_alignment(ref_gray, target_gray)
        if conf > 0.1:
            results.append(('Centroid Alignment', shift_x, shift_y, conf))
            log_debug(f"重心对齐: 偏移({shift_x:.1f}, {shift_y:.1f}), 置信度={conf:.3f}")
        else:
            log_debug("重心对齐算法失败")

    if results:
        results.sort(key=lambda x: x[3], reverse=True)
        best_method, best_x, best_y, best_conf = results[0]
        log_debug(f"选择最佳结果: {best_method}")
        return best_x, best_y, best_conf, best_method
    
    log_debug("所有IMPPG算法都失败，将回退到圆心对齐")
    return None, None, 0, "All methods failed"

# ----------------- 稳健外缘RANSAC与遮罩相位相关（新增） -----------------

def _fit_circle_least_squares(points):
    """最小二乘圆拟合（Pratt/Taubin 简化版）"""
    if len(points) < 3:
        return None
    A = np.c_[2*points[:,0], 2*points[:,1], np.ones(points.shape[0])]
    b = points[:,0]**2 + points[:,1]**2
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = x[0], x[1]
        r = math.sqrt(x[2] + cx*cx + cy*cy)
        return (float(cx), float(cy), float(r))
    except Exception:
        return None

def _fit_circle_ransac(points, iterations=120, threshold=2.0, min_inliers=40):
    """RANSAC 圆拟合，适合混有噪声/错误边缘点的情况"""
    if len(points) < 3:
        return None
    best_circle = None
    best_inliers = 0
    N = len(points)
    for _ in range(iterations):
        try:
            idx = np.random.choice(N, 3, replace=False)
        except ValueError:
            return None
        tri = points[idx]
        cand = _fit_circle_least_squares(tri)
        if cand is None:
            continue
        cx, cy, r = cand
        d = np.sqrt((points[:,0]-cx)**2 + (points[:,1]-cy)**2)
        inliers = np.sum(np.abs(d - r) < threshold)
        if inliers > best_inliers:
            best_inliers = inliers
            mask = (np.abs(d - r) < threshold)
            best_circle = _fit_circle_least_squares(points[mask])
    if best_circle is not None and best_inliers >= min_inliers:
        return best_circle
    return None

def _edge_points_outer_rim(gray, prev_circle=None):
    """
    提取更可靠的“外缘”边界点：
    - 先 Canny
    - 如有上一帧圆，限定在半径的窄环带（r*0.85~1.15）内搜边
    - 方向性筛选：梯度大致指向外法向
    """
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    if prev_circle is not None:
        cx, cy, r = prev_circle
        d = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)
        ring_mask = (d > r*0.85) & (d < r*1.15)
        pts = pts[ring_mask]
        if len(pts) == 0:
            return None

        # 方向性筛选：梯度方向与从中心指向边缘的外法向基本一致
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        keep = []
        for x, y in pts:
            vx, vy = x - cx, y - cy
            nrm = math.hypot(vx, vy) + 1e-6
            nx, ny = vx/nrm, vy/nrm       # 外法向
            gxx = gx[int(y), int(x)]
            gyy = gy[int(y), int(x)]
            gn = math.hypot(gxx, gyy) + 1e-6
            gx_n, gy_n = gxx/gn, gyy/gn   # 梯度方向
            if (gx_n*nx + gy_n*ny) > 0.2:
                keep.append([x, y])
        if len(keep) >= 30:
            pts = np.asarray(keep, dtype=np.float32)
        elif len(pts) < 30:
            return None

    return pts

def detect_circle_robust(gray, prev_circle=None):
    """
    稳健圆检测：外缘方向 + 环带 ROI + RANSAC。
    - 若外缘角度覆盖不足（<120°），锁定半径仅估中心。
    - 若失败，返回 prev_circle（由上层回退到旧路径）。
    """
    pts = _edge_points_outer_rim(gray, prev_circle)
    if pts is None or len(pts) < 30:
        return prev_circle

    cand = _fit_circle_ransac(pts)
    if cand is None:
        return prev_circle

    cx, cy, r = cand

    # 角度覆盖评估（粗略）
    vec = np.arctan2(pts[:,1]-cy, pts[:,0]-cx)
    span = np.ptp(vec)
    if prev_circle is not None and span < (2*np.pi/3.0):  # <120°
        # 残月：锁半径
        cx, cy, _ = cand
        cx_prev, cy_prev, r_prev = prev_circle
        cand = (cx, cy, r_prev)

    return (float(cand[0]), float(cand[1]), float(cand[2]))

def masked_phase_corr(ref_gray, tgt_gray, cx, cy, r):
    """
    只在月盘内做相位相关，忽略背景与阴影边界，得到亚像素平移。
    返回 (dx, dy)，表示将目标再平移多少对齐到参考。
    """
    H, W = ref_gray.shape
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    mask = (dist <= r*0.98).astype(np.float32)
    # 柔边（汉宁窗）以减少边界伪峰
    band = (dist >= r*0.90) & (dist <= r*0.98)
    t = (dist[band] - r*0.90) / (r*0.08 + 1e-6)
    mask[band] = 0.5*(1 + np.cos(np.pi*(1 - t)))

    rg = (ref_gray * mask).astype(np.float32)
    tg = (tgt_gray * mask).astype(np.float32)

    (dx, dy), _ = cv2.phaseCorrelate(rg, tg)
    return float(dx), float(dy)

# ----------------- PHD2增强圆检测算法 -----------------

def adaptive_preprocessing(image, brightness_mode="auto"):
    """自适应预处理 - 优化内存使用"""
    try:
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        mean_brightness = np.mean(gray)

        if brightness_mode == "auto":
            if mean_brightness > 140:
                brightness_mode = "bright"
            elif mean_brightness < 70:
                brightness_mode = "dark"
            else:
                brightness_mode = "normal"

        if brightness_mode == "bright":
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        elif brightness_mode == "dark":
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return filtered, brightness_mode
    except Exception as e:
        print(f"预处理失败: {e}")
        return gray if 'gray' in locals() else image, "error"

def evaluate_circle_quality(image, circle):
    """圆质量评估 - 优化性能"""
    try:
        cx, cy, radius = int(circle[0]), int(circle[1]), int(circle[2])
        h, w = image.shape[:2]

        if (cx - radius < 5 or cy - radius < 5 or
            cx + radius >= w - 5 or cy + radius >= h - 5):
            return 0

        angles = np.linspace(0, 2 * np.pi, 48)
        edge_strengths = []

        for angle in angles:
            inner_x = int(cx + (radius - 2) * np.cos(angle))
            inner_y = int(cy + (radius - 2) * np.sin(angle))
            outer_x = int(cx + (radius + 2) * np.cos(angle))
            outer_y = int(cy + (radius + 2) * np.sin(angle))

            if (0 <= inner_x < w and 0 <= inner_y < h and
                0 <= outer_x < w and 0 <= outer_y < h):

                inner_val = float(image[inner_y, inner_x])
                outer_val = float(image[outer_y, outer_x])
                edge_strength = abs(outer_val - inner_val)
                edge_strengths.append(edge_strength)

        if not edge_strengths:
            return 0

        avg_edge_strength = np.mean(edge_strengths)
        consistency = 1.0 / (1.0 + np.std(edge_strengths) / max(1.0, avg_edge_strength))
        quality_score = avg_edge_strength * consistency
        return min(100.0, quality_score)

    except Exception as e:
        print(f"质量评估失败: {e}")
        return 0

def detect_circle_phd2_enhanced(image, min_radius, max_radius, param1, param2):
    """增强的圆检测 - 内存优化版本"""
    try:
        processed, brightness_mode = adaptive_preprocessing(image, "auto")
        
        best_circle = None
        best_score = 0
        detection_method = "none"

        # 先尝试：稳健外缘RANSAC（优先，失败则走原有路径）
        try:
            robust = detect_circle_robust(processed, None)
            if robust is not None:
                quality = evaluate_circle_quality(processed, robust) * 1.05  # 略微偏置到稳健解
                if quality > best_score:
                    best_score = quality
                    best_circle = np.array(robust, dtype=np.float32)
                    detection_method = "稳健外缘RANSAC"
        except Exception:
            pass

        # 方法1: 标准霍夫圆检测
        try:
            height, _ = processed.shape
            circles = cv2.HoughCircles(
                processed, cv2.HOUGH_GRADIENT,
                dp=1, minDist=height,
                param1=param1, param2=param2,
                minRadius=min_radius, maxRadius=max_radius
            )

            if circles is not None:
                for circle in circles[0]:
                    quality = evaluate_circle_quality(processed, circle)
                    if quality > best_score:
                        best_score = quality
                        best_circle = circle
                        detection_method = f"标准霍夫(P1={param1},P2={param2})"
        except Exception:
            pass

        # 方法2: 自适应参数调整
        if best_score < 15:
            try:
                if brightness_mode == "bright":
                    alt_param1, alt_param2 = param1 + 20, max(param2 - 5, 10)
                elif brightness_mode == "dark":
                    alt_param1, alt_param2 = max(param1 - 15, 20), max(param2 - 10, 5)
                else:
                    alt_param1, alt_param2 = param1, max(param2 - 8, 8)

                circles2 = cv2.HoughCircles(
                    processed, cv2.HOUGH_GRADIENT,
                    dp=1.2, minDist=height // 2,
                    param1=alt_param1, param2=alt_param2,
                    minRadius=min_radius, maxRadius=max_radius
                )

                if circles2 is not None:
                    for circle in circles2[0]:
                        quality = evaluate_circle_quality(processed, circle)
                        if quality > best_score:
                            best_score = quality
                            best_circle = circle
                            detection_method = f"自适应霍夫(P1={alt_param1},P2={alt_param2})"
            except Exception:
                pass

        # 方法3: 轮廓检测作为备选
        if best_score < 10:
            try:
                mean_val = np.mean(processed)
                thresh_val = max(50, int(mean_val * 0.7))

                _, binary = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if min_radius ** 2 * np.pi * 0.3 <= area <= max_radius ** 2 * np.pi * 2.0:
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)
                        if min_radius <= radius <= max_radius:
                            circle = np.array([cx, cy, radius])
                            quality = evaluate_circle_quality(processed, circle) * 0.7
                            if quality > best_score:
                                best_score = quality
                                best_circle = circle
                                detection_method = f"轮廓检测(T={thresh_val})"
            except Exception:
                pass

        return best_circle, processed, best_score, detection_method, brightness_mode
    
    except Exception as e:
        print(f"圆检测失败: {e}")
        return None, image, 0, "error", "unknown"

# ----------------- 集成对齐算法 -----------------
def align_moon_images_incremental(input_folder, output_folder, hough_params,
                                 log_box=None, debug_mode=False, debug_image_basename="",
                                 completion_callback=None, progress_callback=None, 
                                 reference_image_path=None, use_advanced_alignment=False,
                                 alignment_method='auto'):
    """增量处理版本的月球对齐算法 - 边检测边对齐边保存"""
    memory_manager = MemoryManager()
    
    try:
        input_folder = normalize_path(input_folder)
        output_folder = normalize_path(output_folder)
        
        if not ensure_dir_exists(output_folder):
            raise Exception(f"无法创建输出文件夹: {output_folder}")

        debug_output_folder = safe_join(output_folder, "debug")
        if debug_mode and not ensure_dir_exists(debug_output_folder):
            raise Exception(f"无法创建调试文件夹: {debug_output_folder}")

        try:
            image_files = sorted([f for f in os.listdir(input_folder)
                                  if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS])
        except Exception as e:
            raise Exception(f"读取输入文件夹失败: {e}")

        if not image_files:
            raise Exception(f"在 '{input_folder}' 中未找到支持的图片文件")

        min_rad, max_rad, param1, param2 = hough_params
        total_files = len(image_files)
        
        log("=" * 60, log_box)
        log(f"月食圆面对齐工具 V{VERSION} - 增量处理版", log_box)
        log(f"处理模式: 增量处理 (边检测边保存)", log_box)
        log(f"文件总数: {total_files}", log_box)
        log(f"IMPPG高级算法: {'启用' if use_advanced_alignment else '禁用'}", log_box)
        log("=" * 60, log_box)

        # 第一阶段：确定参考图像
        log("阶段 1/2: 确定参考图像...", log_box)
        
        reference_image = None
        reference_center = None
        reference_filename = None
        best_quality = 0
        reference_radius = None
        
        # 如果用户指定了参考图像，优先使用
        if reference_image_path and os.path.exists(reference_image_path):
            ref_filename = os.path.basename(reference_image_path)
            if ref_filename in image_files:
                log(f"加载用户指定的参考图像: {ref_filename}", log_box)
                ref_img = imread_unicode(reference_image_path, cv2.IMREAD_UNCHANGED)
                if ref_img is not None:
                    circle, _, quality, method, brightness = detect_circle_phd2_enhanced(
                        ref_img, min_rad, max_rad, param1, param2
                    )
                    if circle is not None:
                        reference_image = ref_img.copy()
                        reference_center = (circle[0], circle[1])
                        reference_filename = ref_filename
                        best_quality = quality
                        reference_radius = circle[2]
                        log(f"✓ 参考图像检测成功: 质量={quality:.1f}, 方法={method}", log_box)
                    else:
                        log(f"✗ 参考图像检测失败，将自动选择", log_box)
        
        # 如果没有有效的参考图像，需要快速扫描找到最佳的
        if reference_image is None:
            log("自动选择参考图像 (快速扫描前10张)...", log_box)
            scan_count = min(10, total_files)  # 只扫描前10张图片
            
            for i, filename in enumerate(image_files[:scan_count]):
                if progress_callback:
                    progress = int((i / scan_count) * 20)  # 前20%进度用于参考图像选择
                    progress_callback(progress, f"扫描参考图像: {filename}")
                
                input_path = safe_join(input_folder, filename)
                image_original = imread_unicode(input_path, cv2.IMREAD_UNCHANGED)
                
                if image_original is None:
                    continue
                
                circle, _, quality, method, brightness = detect_circle_phd2_enhanced(
                    image_original, min_rad, max_rad, param1, param2
                )
                
                if circle is not None and quality > best_quality:
                    # 释放之前的参考图像
                    if reference_image is not None:
                        del reference_image
                    
                    reference_image = image_original.copy()
                    reference_center = (circle[0], circle[1])
                    reference_filename = filename
                    best_quality = quality
                    reference_radius = circle[2]
                    log(f"  候选参考图像: {filename}, 质量={quality:.1f}", log_box)
                
                del image_original
                force_garbage_collection()
        
        if reference_image is None:
            raise Exception("无法找到有效的参考图像，请检查图像质量和参数设置")
        
        log(f"🎯 最终参考图像: {reference_filename}, 质量评分={best_quality:.1f}", log_box)

        # 第二阶段：增量处理所有图像
        log(f"\n阶段 2/2: 增量处理所有图像...", log_box)
        
        success_count = 0
        failed_files = []
        brightness_stats = {"bright": 0, "normal": 0, "dark": 0}
        method_stats = {}
        
        for i, filename in enumerate(image_files):
            if progress_callback:
                progress = 20 + int((i / total_files) * 80)  # 剩余80%进度用于处理
                progress_callback(progress, f"处理: {filename}")
            
            try:
                input_path = safe_join(input_folder, filename)
                
                # 处理参考图像
                if filename == reference_filename:
                    # 直接保存参考图像，无需变换
                    output_path = safe_join(output_folder, f"aligned_{filename}")
                    if imwrite_unicode(output_path, reference_image):
                        success_count += 1
                        log(f"  🎯 {filename}: [参考图像] 已保存", log_box)
                        
                        # 处理调试图像
                        if debug_mode and filename == debug_image_basename:
                            self._save_debug_image(reference_image, reference_center, 
                                                 reference_center, 0, 0, 1.0, "Reference Image",
                                                 debug_output_folder, filename, reference_filename)
                    else:
                        log(f"  ✗ {filename}: 保存失败", log_box)
                        failed_files.append(filename)
                    continue
                
                # 加载目标图像
                target_image = imread_unicode(input_path, cv2.IMREAD_UNCHANGED)
                if target_image is None:
                    log(f"  ✗ {filename}: 读取失败", log_box)
                    failed_files.append(filename)
                    continue
                
                # 检测圆心
                circle, processed, quality, method, brightness = detect_circle_phd2_enhanced(
                    target_image, min_rad, max_rad, param1, param2
                )
                
                if circle is None:
                    log(f"  ✗ {filename}: 圆检测失败", log_box)
                    failed_files.append(filename)
                    del target_image
                    continue
                
                # 统计信息
                brightness_stats[brightness] += 1
                method_stats[method] = method_stats.get(method, 0) + 1
                
                target_center = (circle[0], circle[1])
                
                # 计算对齐偏移（圆心对齐路径）
                shift_x = reference_center[0] - target_center[0]
                shift_y = reference_center[1] - target_center[1]

                # 用圆质量分映射为置信度（不启用 IMPPG 时）
                confidence = max(0.30, min(0.98, quality / 100.0))
                align_method = "Circle Center"
                
                if use_advanced_alignment and alignment_method != 'circle_only':
                    # 使用IMPPG高级算法
                    adv_shift_x, adv_shift_y, adv_confidence, adv_method = multi_method_alignment(
                        reference_image, target_image, alignment_method,
                        lambda msg: log(f"      {msg}", log_box)
                    )
                    
                    # 如果高级算法成功，使用其结果
                    if adv_shift_x is not None and adv_confidence > 0.2:
                        shift_x, shift_y = adv_shift_x, adv_shift_y
                        confidence = adv_confidence
                        align_method = adv_method
                    else:
                        # 回退到圆心对齐
                        shift_x = reference_center[0] - target_center[0]
                        shift_y = reference_center[1] - target_center[1]
                        align_method = "Circle Center (Fallback)"
                else:
                    # 传统圆心对齐
                    shift_x = reference_center[0] - target_center[0]
                    shift_y = reference_center[1] - target_center[1]
                
                # 应用变换
                rows, cols = target_image.shape[:2]
                translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                aligned_image = cv2.warpAffine(
                    target_image, translation_matrix, (cols, rows),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                # 亚像素微调：遮罩相位相关（仅在参考圆存在时）
                try:
                    if reference_radius is not None:
                        ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY) if len(reference_image.shape) > 2 else reference_image
                        tgt_gray2 = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY) if len(aligned_image.shape) > 2 else aligned_image
                        dx2, dy2 = masked_phase_corr(ref_gray, tgt_gray2, float(reference_center[0]), float(reference_center[1]), float(reference_radius))
                        if abs(dx2) > 1e-3 or abs(dy2) > 1e-3:
                            M2 = np.float32([[1, 0, dx2], [0, 1, dy2]])
                            aligned_image = cv2.warpAffine(aligned_image, M2, (cols, rows),
                                                           flags=cv2.INTER_LANCZOS4,
                                                           borderMode=cv2.BORDER_CONSTANT,
                                                           borderValue=0)
                except Exception:
                    pass
                
                # 立即保存对齐后的图像
                output_path = safe_join(output_folder, f"aligned_{filename}")
                if imwrite_unicode(output_path, aligned_image):
                    success_count += 1
                    log(f"  ✓ {filename}: 偏移=({shift_x:.1f}, {shift_y:.1f}), "
                       f"置信度={confidence:.3f}, 质量={quality:.1f} - 已保存", log_box)
                    
                    # 处理调试图像
                    if debug_mode and filename == debug_image_basename and processed is not None:
                        self._save_debug_image(processed, target_center, reference_center,
                                             shift_x, shift_y, confidence, align_method,
                                             debug_output_folder, filename, reference_filename)
                else:
                    log(f"  ✗ {filename}: 变换成功但保存失败", log_box)
                    failed_files.append(filename)
                
                # 立即释放内存
                del target_image, aligned_image
                if 'processed' in locals():
                    del processed
                force_garbage_collection()
                
            except Exception as e:
                log(f"  ✗ {filename}: 处理异常 - {e}", log_box)
                failed_files.append(filename)
                # 确保内存被释放
                for var_name in ['target_image', 'aligned_image', 'processed']:
                    if var_name in locals():
                        del locals()[var_name]
                force_garbage_collection()

        if progress_callback:
            progress_callback(100, "处理完成")

        # 清理参考图像
        del reference_image
        force_garbage_collection()

        # 输出统计信息
        log("=" * 60, log_box)
        log(f"增量对齐完成! 成功对齐 {success_count}/{total_files} 张图像", log_box)
        log(f"使用参考图像: {reference_filename}", log_box)
        log(f"对齐算法: {'IMPPG高级算法' if use_advanced_alignment else 'PHD2圆心算法'}", log_box)
        
        if failed_files:
            log(f"失败文件({len(failed_files)}): {', '.join(failed_files[:5])}" + 
               ("..." if len(failed_files) > 5 else ""), log_box)
        
        log(f"亮度分布: 明亮={brightness_stats['bright']}, 正常={brightness_stats['normal']}, 暗={brightness_stats['dark']}", log_box)
        if method_stats:
            log(f"检测方法: {', '.join([f'{k}={v}' for k, v in method_stats.items()])}", log_box)
        
        log(f"当前内存使用: {get_memory_usage_mb():.1f} MB", log_box)
        
        if completion_callback:
            completion_callback(True, f"增量处理完成！成功对齐 {success_count}/{total_files} 张图像")

    except Exception as e:
        import traceback
        error_msg = f"增量处理过程中发生错误: {e}\n{traceback.format_exc()}"
        log(error_msg, log_box)
        if completion_callback:
            completion_callback(False, error_msg)
    finally:
        force_garbage_collection()

def _save_debug_image(self, processed_img, target_center, reference_center, 
                     shift_x, shift_y, confidence, method, 
                     debug_output_folder, filename, reference_filename):
    """保存调试图像的辅助函数"""
    try:
        if processed_img is None:
            return
            
        # 转换为BGR用于绘制
        if len(processed_img.shape) == 2:
            debug_image = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        else:
            debug_image = processed_img.copy()
        
        # 绘制检测结果
        # 检测到的圆心 (红色)
        cv2.circle(debug_image, (int(target_center[0]), int(target_center[1])), 5, (0, 0, 255), -1)
        # 参考圆心位置 (黄色)
        cv2.circle(debug_image, (int(reference_center[0]), int(reference_center[1])), 15, (0, 255, 255), 3)
        # 连接线
        cv2.line(debug_image, (int(target_center[0]), int(target_center[1])),
                 (int(reference_center[0]), int(reference_center[1])), (0, 255, 255), 2)

        # 添加文本信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        texts = [
            f"Method: {method[:35]}",
            f"Shift: ({shift_x:.1f}, {shift_y:.1f})",
            f"Confidence: {confidence:.3f}",
            f"Reference: {reference_filename}",
            f"Mode: Incremental Processing"
        ]
        
        for j, text in enumerate(texts):
            cv2.putText(debug_image, text, (10, 25 + j * 25),
                        font, font_scale, (255, 255, 255), thickness)

        debug_path = safe_join(debug_output_folder, f"debug_{filename}")
        imwrite_unicode(debug_path, debug_image)
        
    except Exception as e:
        print(f"调试图像生成失败: {e}")

def log(msg, log_box=None):
    """跨平台日志输出"""
    if log_box:
        try:
            log_box.master.after(0, lambda: (
                log_box.config(state="normal"),
                log_box.insert(tk.END, str(msg) + "\n"),
                log_box.see(tk.END),
                log_box.config(state="disabled")
            ))
        except Exception:
            pass
    if msg:
        print(msg)

# ---------- 优化的UI部分 ----------

class PreviewWindow(tk.Toplevel):
    """预览窗口 - 跨平台优化版本"""
    def __init__(self, master, app_controller):
        super().__init__(master)
        self.app = app_controller
        self.title("预览与半径估计")
        self.geometry("900x600")
        self.minsize(600, 400)
        
        self.center_window()
        self.configure_fonts()
        
        self.preview_img_cv = None
        self.preview_img_disp = None
        self.current_preview_path = None
        self.preview_scale = 1.0
        self.rect_state = {"start": None, "rect": None}
        self.delta_var = tk.IntVar(value=100)
        self.estimate_radius_px = tk.IntVar(value=0)
        
        self._create_widgets()
        self._bind_events()
        self._show_initial_hint()

    def center_window(self):
        """跨平台窗口居中"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def configure_fonts(self):
        """配置跨平台字体"""
        try:
            self.default_font = DEFAULT_FONT
        except Exception:
            self.default_font = ("TkDefaultFont", 9)

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # 工具栏
        tool_frame = ttk.Frame(main_frame)
        tool_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Button(tool_frame, text="选择样张", command=self.choose_preview_image).pack(side="left", padx=(0, 10))
        ttk.Label(tool_frame, text="增减范围 Δ:").pack(side="left", padx=(10, 5))
        ttk.Spinbox(tool_frame, from_=0, to=5000, textvariable=self.delta_var, width=8).pack(side="left")
        
        self.est_label = ttk.Label(tool_frame, text=" | 估计半径: —")
        self.est_label.pack(side="left", padx=(10, 5))
        
        apply_btn = ttk.Button(tool_frame, text="✔ 应用到主界面", command=self.apply_to_main_sliders)
        apply_btn.pack(side="right")
        
        # 画布
        self.canvas = tk.Canvas(main_frame, background="#333", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")

    def _bind_events(self):
        self.canvas.bind("<Configure>", self._render_preview)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

    def _show_initial_hint(self):
        self.canvas.delete("hint")
        self.canvas.update_idletasks()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            self.after(100, self._show_initial_hint)
            return
        
        font_size = 20 if IS_MACOS else 16
        self.canvas.create_text(cw / 2, ch / 2, 
                                text="请选择样张，在图上拖拽鼠标框选月亮", 
                                font=(self.default_font[0], font_size), 
                                fill="gray60", tags="hint")

    def choose_preview_image(self):
        initdir = self.app.input_var.get() if os.path.isdir(self.app.input_var.get()) else os.getcwd()
        
        filetypes = [("支持的图像", " ".join(f"*{ext}" for ext in SUPPORTED_EXTS)), ("所有文件", "*.*")]
        
        path = filedialog.askopenfilename(
            title="选择样张用于预览与框选",
            filetypes=filetypes,
            initialdir=initdir, 
            parent=self
        )
        
        if not path:
            return
            
        path = normalize_path(path)
        img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            messagebox.showerror("错误", "无法读取该图像。", parent=self)
            return
            
        self.preview_img_cv = to_display_rgb(img)
        self.current_preview_path = path
        self.title(f"预览与半径估计 - {os.path.basename(path)}")
        self._render_preview()

    def _render_preview(self, event=None):
        self.canvas.delete("all")
        if self.preview_img_cv is None:
            self._show_initial_hint()
            return
            
        h, w = self.preview_img_cv.shape[:2]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.preview_scale = min(cw / w, ch / h, 1.0)
        new_w, new_h = int(w * self.preview_scale), int(h * self.preview_scale)
        
        try:
            disp = Image.fromarray(self.preview_img_cv).resize((new_w, new_h), Image.LANCZOS)
            self.preview_img_disp = ImageTk.PhotoImage(disp)
            self.canvas.create_image(cw / 2, ch / 2, image=self.preview_img_disp, anchor="center")
        except Exception as e:
            print(f"图像显示失败: {e}")
            
        self.rect_state = {"start": None, "rect": None}
        self.estimate_radius_px.set(0)
        self.est_label.config(text=" | 估计半径: —")

    def _to_image_coords(self, xc, yc):
        """转换画布坐标到图像坐标"""
        if self.preview_img_cv is None:
            return 0, 0
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        h, w = self.preview_img_cv.shape[:2]
        new_w, new_h = int(w * self.preview_scale), int(h * self.preview_scale)
        ox, oy = (cw - new_w) // 2, (ch - new_h) // 2
        xi = int((xc - ox) / self.preview_scale)
        yi = int((yc - oy) / self.preview_scale)
        return max(0, min(w - 1, xi)), max(0, min(h - 1, yi))

    def on_canvas_press(self, event):
        self.canvas.delete("hint")
        if self.preview_img_cv is None:
            return
        if self.rect_state["rect"]:
            self.canvas.delete(self.rect_state["rect"])
            self.rect_state["rect"] = None
        self.rect_state["start"] = (event.x, event.y)

    def on_canvas_drag(self, event):
        if self.rect_state["start"] is None:
            return
        x0, y0 = self.rect_state["start"]
        if self.rect_state["rect"] is None:
            self.rect_state["rect"] = self.canvas.create_rectangle(
                x0, y0, event.x, event.y, outline="#00BFFF", width=2
            )
        else:
            self.canvas.coords(self.rect_state["rect"], x0, y0, event.x, event.y)

    def on_canvas_release(self, event):
        if self.rect_state["start"] is None:
            return
        x0, y0 = self.rect_state["start"]
        xi0, yi0 = self._to_image_coords(x0, y0)
        xi1, yi1 = self._to_image_coords(event.x, event.y)
        w_px, h_px = abs(xi1 - xi0), abs(yi1 - yi0)
        self.rect_state["start"] = None
        
        if w_px < 4 or h_px < 4:
            if self.rect_state["rect"]:
                self.canvas.delete(self.rect_state["rect"])
                self.rect_state["rect"] = None
            return
            
        radius = int(min(w_px, h_px) / 2)
        self.estimate_radius_px.set(radius)
        self.est_label.config(text=f" | 估计半径: {radius} px")

    def apply_to_main_sliders(self):
        r = self.estimate_radius_px.get()
        if r <= 0:
            messagebox.showwarning("提示", "请先在图像上框选一个月球区域来估计半径。", parent=self)
            return

        d = max(0, self.delta_var.get())
        min_r = max(1, r - d)
        max_r = max(min_r + 1, r + d)

        self.app.params["min_radius"].set(min_r)
        self.app.params["max_radius"].set(max_r)

        # 询问是否将当前预览图像设为参考图像
        if hasattr(self, 'current_preview_path') and self.current_preview_path:
            use_as_ref = messagebox.askyesno("设置参考图像", 
                                           f"是否将当前预览的图像设为参考图像？\n\n"
                                           f"图像: {os.path.basename(self.current_preview_path)}\n"
                                           f"估计半径: {r} px", 
                                           parent=self)
            if use_as_ref:
                self.app.reference_image_var.set(self.current_preview_path)

        messagebox.showinfo("成功", f"半径范围已成功更新为:\nMin: {min_r}\nMax: {max_r}", parent=self)

class ProgressWindow(tk.Toplevel):
    """进度显示窗口"""
    def __init__(self, master):
        super().__init__(master)
        self.title("处理进度")
        self.geometry("400x150")
        self.resizable(False, False)
        
        self.transient(master)
        self.grab_set()
        
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        self.status_label = ttk.Label(main_frame, text="准备开始...", font=UI_FONT)
        self.status_label.pack(pady=(0, 10))
        
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=(0, 10))
        
        self.percent_label = ttk.Label(main_frame, text="0%", font=UI_FONT)
        self.percent_label.pack()
        
        self.center_window()
    
    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
    
    def update_progress(self, progress, status):
        self.progress_var.set(progress)
        self.status_label.config(text=status)
        self.percent_label.config(text=f"{progress}%")
        self.update()

class UniversalLunarAlignApp:
    """主应用程序 - 集成版"""
    def __init__(self, root):
        self.root = root
        self.root.title(f"月食圆面对齐工具 V{VERSION} By @正七价的氟离子")
        self.root.geometry("920x800")
        self.root.minsize(750, 700)
        
        self.setup_cross_platform()
        
        self.preview_window = None
        self.progress_window = None
        
        self._init_vars()
        self._create_main_layout()
        self._create_path_widgets()
        self._create_param_widgets()
        self._create_imppg_widgets()
        self._create_debug_widgets()
        self._create_action_widgets()
        self._create_log_widgets()
        self._set_initial_log_message()
        self.on_debug_mode_change()
        self.on_advanced_change()

    def setup_cross_platform(self):
        """跨平台设置"""
        try:
            if IS_WINDOWS:
                self.root.iconbitmap(default='')
        except Exception:
            pass
        
        style = ttk.Style()
        try:
            if IS_WINDOWS:
                style.theme_use('winnative')
            elif IS_MACOS:
                style.theme_use('aqua')
            else:
                style.theme_use('clam')
        except Exception:
            pass

    def _init_vars(self):
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.reference_image_var = tk.StringVar()
        self.debug_var = tk.BooleanVar(value=DEFAULT_DEBUG_MODE)
        self.debug_image_path_var = tk.StringVar(value="")
        self.params = {
            "min_radius": tk.IntVar(value=300),
            "max_radius": tk.IntVar(value=800),
            "param1": tk.IntVar(value=50),
            "param2": tk.IntVar(value=30)
        }
        
        # IMPPG相关变量
        self.use_advanced_alignment = tk.BooleanVar(value=False)
        self.alignment_method = tk.StringVar(value="auto")

    def _create_main_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        
        # 控制面板
        control_pane = ttk.Frame(self.root, padding=(10, 10, 10, 0))
        control_pane.grid(row=0, column=0, sticky="ew")
        control_pane.columnconfigure(0, weight=1)
        control_pane.columnconfigure(1, weight=0)
        
        # 路径设置框架
        self.path_frame = ttk.LabelFrame(control_pane, text="1. 路径设置", padding=10)
        self.path_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        
        # 参数设置框架
        params_container = ttk.Frame(control_pane)
        params_container.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        params_container.columnconfigure(0, weight=2)
        params_container.columnconfigure(1, weight=1)
        
        self.param_frame = ttk.LabelFrame(params_container, text="2. PHD2霍夫圆参数", padding=10)
        self.param_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # IMPPG算法框架
        self.imppg_frame = ttk.LabelFrame(params_container, text="3. IMPPG高级算法", padding=10)
        self.imppg_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # 调试框架
        self.debug_frame = ttk.LabelFrame(control_pane, text="4. 预览与调试", padding=10)
        self.debug_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        # 操作框架
        self.action_frame = ttk.Frame(self.root, padding=(0, 10))
        self.action_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _create_path_widgets(self):
        frame = self.path_frame
        frame.columnconfigure(1, weight=1)
        
        # 输入文件夹
        ttk.Label(frame, text="输入文件夹:", font=UI_FONT).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.input_var, font=UI_FONT).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="浏览...", command=self.select_input_folder).grid(row=0, column=2, padx=5, pady=5)
        
        # 输出文件夹
        ttk.Label(frame, text="输出文件夹:", font=UI_FONT).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.output_var, font=UI_FONT).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="浏览...", command=self.select_output_folder).grid(row=1, column=2, padx=5, pady=5)
        
        # 参考图像选择
        ttk.Label(frame, text="参考图像:", font=UI_FONT).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.reference_entry = ttk.Entry(frame, textvariable=self.reference_image_var, font=UI_FONT)
        self.reference_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        ref_btn_frame = ttk.Frame(frame)
        ref_btn_frame.grid(row=2, column=2, padx=5, pady=5)
        
        ttk.Button(ref_btn_frame, text="选择", command=self.select_reference_image).pack(side="left")
        ttk.Button(ref_btn_frame, text="清除", command=self.clear_reference_image).pack(side="left", padx=(2,0))
        
        # 提示文本
        help_text = ttk.Label(frame, text="💡参考图像：作为对齐基准的图像。请在预览&半径估计窗口选择。", 
                              font=(UI_FONT[0], UI_FONT[1]-1), foreground="gray")
        help_text.grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=(2,5))

    def _create_param_widgets(self):
        frame = self.param_frame
        frame.columnconfigure(1, weight=1)
        
        # 帮助文本
        help_text = ("• PHD2增强算法：三级检测策略，自适应图像亮度\n"
                     "• 最小/最大半径: 限制检测到的圆的半径范围(像素)\n"
                     "• 参数1: Canny边缘检测高阈值\n"
                     "• 参数2: 霍夫累加器阈值（关键参数）")
        
        help_label = ttk.Label(frame, text=help_text, justify="left", font=(UI_FONT[0], UI_FONT[1]-1))
        help_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 10))

        # 参数控件
        param_defs = [
            ("min_radius", "最小半径:", 1, 3000),
            ("max_radius", "最大半径:", 10, 4000),
            ("param1", "参数1 (Canny):", 1, 200),
            ("param2", "参数2 (累加阈值):", 1, 100)
        ]
        
        for i, (key, label, min_val, max_val) in enumerate(param_defs):
            row_index = i + 1
            var = self.params[key]
            
            ttk.Label(frame, text=label, font=UI_FONT).grid(row=row_index, column=0, sticky="w", padx=5, pady=3)
            ttk.Scale(frame, from_=min_val, to=max_val, orient="horizontal", variable=var,
                      command=lambda v, k=key: self.params[k].set(int(float(v)))).grid(row=row_index, column=1, sticky="ew", padx=5, pady=3)
            ttk.Spinbox(frame, from_=min_val, to=max_val, textvariable=var, width=6, font=UI_FONT).grid(row=row_index, column=2, padx=5, pady=3)

    def _create_imppg_widgets(self):
        frame = self.imppg_frame
        
        # 启用IMPPG算法复选框
        cb_advanced = ttk.Checkbutton(frame, text="启用IMPPG算法", 
                                     variable=self.use_advanced_alignment,
                                     command=self.on_advanced_change)
        cb_advanced.pack(fill="x", padx=5, pady=(0, 10))

        # 算法选择
        ttk.Label(frame, text="算法类型:", font=UI_FONT).pack(anchor="w", padx=5)
        self.method_combo = ttk.Combobox(frame, textvariable=self.alignment_method,
                                        values=['auto', 'phase_corr', 'template', 'feature', 'centroid'],
                                        state="disabled", width=15, font=UI_FONT)
        self.method_combo.pack(fill="x", padx=5, pady=2)
        
        # 算法说明
        algo_help = ("• auto: 自动选择最佳算法\n"
                     "• phase_corr: 相位相关算法\n"
                     "• template: 模板匹配算法\n"
                     "• feature: ORB特征匹配\n"
                     "• centroid: 重心对齐算法")
        
        help_label = ttk.Label(frame, text=algo_help, justify="left", 
                              font=(UI_FONT[0], UI_FONT[1]-2), foreground="darkgreen")
        help_label.pack(anchor="w", padx=5, pady=(5, 10))
        
        # 警告提示
        warning_text = ("⚠️ 实验性功能")
        ttk.Label(frame, text=warning_text, font=(UI_FONT[0], UI_FONT[1]-1), 
                 foreground="orange", justify="center").pack(pady=5)

    def _create_debug_widgets(self):
        frame = self.debug_frame
        frame.columnconfigure(1, weight=1)
        
        # 预览按钮
        ttk.Button(frame, text="打开预览 & 半径估计窗口", 
                   command=self.open_preview).grid(row=0, column=0, columnspan=3, 
                                                   sticky="ew", padx=5, pady=(0, 10))

        # 调试模式复选框
        cb = ttk.Checkbutton(frame, text="启用调试模式", 
                             variable=self.debug_var, 
                             command=self.on_debug_mode_change)
        cb.grid(row=1, column=0, sticky="w", padx=5, pady=5)

        # 调试图像选择
        self.debug_entry = ttk.Entry(frame, textvariable=self.debug_image_path_var, 
                                     state="disabled", font=UI_FONT)
        self.debug_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        self.debug_button = ttk.Button(frame, text="选择调试样张", 
                                       command=self.select_debug_image, 
                                       state="disabled")
        self.debug_button.grid(row=1, column=2, padx=5, pady=5)

    def on_debug_mode_change(self):
        """调试模式开关事件处理"""
        is_enabled = self.debug_var.get()
        new_state = "normal" if is_enabled else "disabled"

        self.debug_entry.config(state=new_state)
        self.debug_button.config(state=new_state)

        if not is_enabled:
            self.debug_image_path_var.set("")

    def on_advanced_change(self):
        """IMPPG算法开关事件处理"""
        is_enabled = self.use_advanced_alignment.get()
        new_state = "readonly" if is_enabled else "disabled"
        self.method_combo.config(state=new_state)

    def _create_action_widgets(self):
        frame = self.action_frame
        frame.columnconfigure(0, weight=1)
        
        # 主操作按钮
        self.start_button = ttk.Button(frame, text="🚀 开始集成对齐", 
                                       command=self.start_alignment)
        self.start_button.pack(pady=10, ipady=8, fill="x", padx=200)
        
        # 设置按钮样式
        try:
            style = ttk.Style()
            style.configure("Accent.TButton", font=(UI_FONT[0], UI_FONT[1] + 1, "bold"))
            self.start_button.configure(style="Accent.TButton")
        except Exception:
            pass

    def _create_log_widgets(self):
        log_pane = ttk.Frame(self.root, padding=(10, 5, 10, 10))
        log_pane.grid(row=2, column=0, columnspan=2, sticky="nsew")
        log_pane.columnconfigure(0, weight=1)
        log_pane.rowconfigure(0, weight=1)
        
        # 日志文本框
        self.log_box = scrolledtext.ScrolledText(log_pane, height=12, wrap="word", 
                                                 relief="solid", borderwidth=1,
                                                 font=UI_FONT)
        self.log_box.pack(fill="both", expand=True)

    def _set_initial_log_message(self):
        """设置初始欢迎消息"""
        scipy_status = "✓ 已安装" if SCIPY_AVAILABLE else "✗ 未安装"
        welcome_message = (f"欢迎使用月食圆面对齐工具 V{VERSION} - 集成版 By @正七价的氟离子\n"
                           f"运行平台: {SYSTEM}\n"
                           f"SciPy状态: {scipy_status}\n"
                           "================================================================\n\n"
                           "算法说明：\n"
                           "• PHD2增强算法：基于霍夫圆检测，适用于完整清晰的月球\n"
                           "• IMPPG高级算法：适用于偏食、生光等复杂阶段（实验性）\n"
                           "• 回退机制：确保在任何情况下都有可用的对齐方案\n\n"
                           "使用建议：\n"
                           "• 完整月食：建议使用PHD2算法（默认）\n"
                           "• 生光阶段：可尝试启用IMPPG算法\n"
                           "• 使用预览工具准确估算半径范围\n"
                           "• 参数2（累加器阈值）是最关键的调整参数\n"
                           "• 启用调试模式可查看详细的检测过程\n"
                           f"• 支持格式：{', '.join(SUPPORTED_EXTS)}\n")
        
        if not SCIPY_AVAILABLE:
            welcome_message += ("\n⚠️ 注意: SciPy未安装，相位相关算法将被禁用\n"
                               "可通过 pip install scipy 安装以启用完整IMPPG功能\n")
        
        self.log_box.insert(tk.END, welcome_message)
        self.log_box.config(state="disabled")

    def select_input_folder(self):
        """选择输入文件夹"""
        path = filedialog.askdirectory(title="选择输入文件夹")
        if path:
            path = normalize_path(path)
            self.input_var.set(path)
            
            # 自动设置输出文件夹
            parent_dir = os.path.dirname(path)
            folder_name = os.path.basename(path)
            output_path = safe_join(parent_dir, f"{folder_name}_aligned_v11")
            self.output_var.set(output_path)

    def select_output_folder(self):
        """选择输出文件夹"""
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_var.set(normalize_path(path))

    def select_reference_image(self):
        """选择参考图像"""
        initdir = self.input_var.get() if os.path.isdir(self.input_var.get()) else os.getcwd()
        
        filetypes = [("支持的图像", " ".join(f"*{ext}" for ext in SUPPORTED_EXTS)), 
                     ("所有文件", "*.*")]
        
        path = filedialog.askopenfilename(
            title="选择参考图像（用作对齐基准）",
            filetypes=filetypes,
            initialdir=initdir
        )
        
        if path:
            path = normalize_path(path)
            input_folder = self.input_var.get().strip()
            if input_folder and not path.startswith(input_folder):
                result = messagebox.askyesno("确认", 
                                           "选择的参考图像不在输入文件夹内。\n"
                                           "建议选择输入文件夹中的图像作为参考。\n"
                                           "是否继续使用此图像？", 
                                           icon='question')
                if not result:
                    return
            
            self.reference_image_var.set(path)

    def clear_reference_image(self):
        """清除参考图像选择"""
        self.reference_image_var.set("")

    def select_debug_image(self):
        """选择调试样张"""
        initdir = self.input_var.get() if os.path.isdir(self.input_var.get()) else os.getcwd()
        
        filetypes = [("支持的图像", " ".join(f"*{ext}" for ext in SUPPORTED_EXTS)), 
                     ("所有文件", "*.*")]
        
        path = filedialog.askopenfilename(
            title="选择调试样张",
            filetypes=filetypes,
            initialdir=initdir
        )
        
        if path:
            self.debug_image_path_var.set(normalize_path(path))

    def open_preview(self):
        """打开预览窗口"""
        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.preview_window = PreviewWindow(self.root, self)
        
        self.preview_window.deiconify()
        self.preview_window.lift()
        self.preview_window.focus_force()

    def show_progress_window(self):
        """显示进度窗口"""
        if self.progress_window is None or not self.progress_window.winfo_exists():
            self.progress_window = ProgressWindow(self.root)
        return self.progress_window

    def start_alignment(self):
        """开始对齐处理"""
        # 输入验证
        in_path = self.input_var.get().strip()
        out_path = self.output_var.get().strip()

        if not os.path.isdir(in_path):
            messagebox.showerror("错误", "请选择有效的输入文件夹。")
            return
            
        if not out_path:
            messagebox.showerror("错误", "请指定输出文件夹。")
            return

        # IMPPG算法验证
        use_advanced = self.use_advanced_alignment.get()
        method = self.alignment_method.get()
        
        if use_advanced and not SCIPY_AVAILABLE and method in ['auto', 'phase_corr']:
            result = messagebox.askyesno("警告", 
                                       "SciPy未安装，相位相关算法将被禁用。\n"
                                       "IMPPG功能可能受限。\n\n"
                                       "是否继续？", 
                                       icon='warning')
            if not result:
                return

        # 参考图像设置
        ref_path = self.reference_image_var.get().strip()
        ref_path = normalize_path(ref_path) if ref_path else None
        
        # 验证参考图像
        if ref_path and not os.path.exists(ref_path):
            result = messagebox.askyesno("警告", 
                                       f"指定的参考图像不存在：\n{ref_path}\n\n"
                                       "是否继续（将自动选择参考图像）？", 
                                       icon='warning')
            if not result:
                return
            ref_path = None

        # 调试设置
        dbg_mode = self.debug_var.get()
        dbg_path = self.debug_image_path_var.get().strip()
        dbg_basename = os.path.basename(dbg_path) if dbg_path else ""

        if dbg_mode and not dbg_basename:
            result = messagebox.askyesno("提示", 
                                         "已开启调试模式，但未选择调试样张。\n"
                                         "处理将继续，但不会生成调试图像。\n"
                                         "是否继续？", 
                                         icon='warning')
            if not result:
                return

        # 获取参数
        hough_params = (
            self.params["min_radius"].get(),
            self.params["max_radius"].get(),
            self.params["param1"].get(),
            self.params["param2"].get()
        )

        # 准备UI
        self.log_box.config(state="normal")
        self.log_box.delete(1.0, tk.END)
        
        # 根据算法类型更新按钮文本
        if use_advanced:
            button_text = "集成对齐中 (IMPPG + PHD2)..."
        else:
            button_text = "PHD2对齐中..."
        
        self.start_button.config(state="disabled", text=button_text)
        
        # 显示进度窗口
        progress_window = self.show_progress_window()

        # 启动处理线程
        def progress_callback(progress, status):
            if progress_window and progress_window.winfo_exists():
                progress_window.update_progress(progress, status)

        threading.Thread(
            target=align_moon_images_incremental,
            args=(in_path, out_path, hough_params, self.log_box, dbg_mode, dbg_basename,
                  self.on_alignment_complete, progress_callback, ref_path, 
                  use_advanced, method),
            daemon=True
        ).start()

    def on_alignment_complete(self, success, message):
        """对齐完成回调"""
        self.root.after(0, lambda: self._update_ui_on_complete(success, message))

    def _update_ui_on_complete(self, success, message):
        """更新UI完成状态"""
        # 恢复按钮
        self.start_button.config(state="normal", text="🚀 开始集成对齐")
        self.log_box.config(state="disabled")
        
        # 关闭进度窗口
        if self.progress_window and self.progress_window.winfo_exists():
            self.progress_window.destroy()
            self.progress_window = None
        
        # 显示结果
        if success:
            messagebox.showinfo("处理完成", message)
        else:
            messagebox.showerror("处理失败", "处理过程中发生错误，详情请查看日志。", detail=message)

def main():
    """主函数"""
    try:
        if ThemedTk is not None:
            if IS_WINDOWS:
                root = ThemedTk(theme="winnative")
            elif IS_MACOS:
                root = ThemedTk(theme="aqua")
            else:
                root = ThemedTk(theme="arc")
        else:
            raise ImportError("ttkthemes not available")
    except Exception as e:
        print(f"主题加载失败，使用默认样式: {e}")
        root = tk.Tk()

    # 创建应用
    app = UniversalLunarAlignApp(root)
    
    # 设置关闭事件
    def on_closing():
        force_garbage_collection()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        force_garbage_collection()

if __name__ == '__main__':
    main()
