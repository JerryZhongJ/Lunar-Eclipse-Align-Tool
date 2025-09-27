# utils_common.py
from pathlib import Path
import os, sys, platform, gc
from typing import Any, NamedTuple
import numpy as np
import cv2
from PIL import Image

import piexif

import tkinter as tk

# ----------------- 系统/常量 -----------------
SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_MACOS = SYSTEM == "Darwin"
IS_LINUX = SYSTEM == "Linux"

DEFAULT_DEBUG_MODE = False
DEFAULT_DEBUG_IMAGE_PATH = ""
SUPPORTED_EXTS = {'*.tif', '*.tiff', '*.bmp', '*.png', '*.jpg', '*.jpeg'}

# 内存管理
MAX_IMAGES_IN_MEMORY = 10
MEMORY_THRESHOLD_MB = 500

def get_memory_usage_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

def force_garbage_collection():
    gc.collect()

class MemoryManager:
    def __init__(self, threshold_mb=MEMORY_THRESHOLD_MB):
        self.threshold_mb = threshold_mb
        self.image_cache = {}
        self.access_order = []
    def should_clear_memory(self):
        return get_memory_usage_mb() > self.threshold_mb or len(self.image_cache) > MAX_IMAGES_IN_MEMORY
    def clear_old_images(self, keep_count=5):
        if len(self.access_order) > keep_count:
            to_remove = self.access_order[:-keep_count]
            for key in to_remove:
                if key in self.image_cache:
                    del self.image_cache[key]
                self.access_order.remove(key)
        force_garbage_collection()

class Hough(NamedTuple):
    min_radius: int
    max_radius: int
    param1: int
    param2: int
    
def imread_unicode(path: Path, flags=cv2.IMREAD_UNCHANGED) ->  np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]] | None:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags)  
    except Exception as e:
        print(f"图像读取失败 {path}: {e}")
        return None

def imwrite_unicode(path: Path, image: np.ndarray)-> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower() or ".tiff"
   
        if ext in (".tif", ".tiff"):
            params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
        else:
            params = []
        ok, buf = cv2.imencode(ext, image, params)
        if not ok:
            raise Exception("图像编码失败")
        buf.tofile(path)
        return True
    except Exception as e:
        print(f"图像保存失败 {path}: {e}")
        return False

def imwrite_with_exif(src_path: Path, dst_path: Path, img_bgr: np.ndarray) -> bool:
    """
    优先保留 src_path 的 EXIF/ICC，用 Pillow 写出 JPEG/TIFF。
    失败或不支持时回退到 imwrite_unicode。
    参数:
        src_path: 原始读取图像的文件路径(用于提取EXIF/ICC)
        dst_path: 目标输出路径
        img_bgr:  OpenCV(BGR) 图像
    返回: bool 是否写出成功
    """
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        ext = dst_path.suffix.lower() or ".tiff"
        # 仅在 JPEG/TIFF 尝试保EXIF，其余格式走原逻辑
        if ext not in (".jpg", ".jpeg", ".tif", ".tiff"):
            return imwrite_unicode(dst_path, img_bgr)

        # Pillow 不可用则直接回退
        if Image is None:
            return imwrite_unicode(dst_path, img_bgr)

        # BGR -> RGB
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            # 如果颜色转换失败，仍尝试直接构建
            rgb = img_bgr
        im = Image.fromarray(rgb)

        exif_bytes = None
        icc = None
        # 读取源图的 EXIF 与 ICC
        try:
            with Image.open(src_path) as src_im:
                exif_bytes = src_im.info.get("exif", None)
                icc = src_im.info.get("icc_profile", None)
                # 将 Orientation 归一到 1，避免查看器再次旋转
                if exif_bytes and piexif is not None:
                    try:
                        exif_dict = piexif.load(exif_bytes)
                        exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
                        exif_bytes = piexif.dump(exif_dict)
                    except Exception:
                        # EXIF 解析失败则保持原样
                        pass
        except Exception:
            pass

        save_kwargs = {}
        if exif_bytes is not None:
            save_kwargs["exif"] = exif_bytes
        if icc is not None:
            save_kwargs["icc_profile"] = icc

        if ext in (".jpg", ".jpeg"):
            # 设一个较高质量，保持文件体积与质量平衡
            save_kwargs.setdefault("quality", 95)
            im.save(dst_path, **save_kwargs)
        else:  # TIFF
            # 无损/轻压缩
            save_kwargs.setdefault("compression", "tiff_deflate")
            im.save(dst_path, **save_kwargs)
        return True
    except Exception:
        # 任意异常回退
        return imwrite_unicode(dst_path, img_bgr)

def to_display_rgb(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2RGB)


# 统一日志
def log(msg, log_box=None):
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
