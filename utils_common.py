# utils_common.py
import os, sys, platform, gc
import numpy as np
import cv2
from PIL import Image
import tkinter as tk

# ----------------- 系统/常量 -----------------
SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_MACOS = SYSTEM == "Darwin"
IS_LINUX = SYSTEM == "Linux"

VERSION = "1.1.4"
DEFAULT_DEBUG_MODE = False
DEFAULT_DEBUG_IMAGE_PATH = ""
SUPPORTED_EXTS = {'.tif', '.tiff', '.bmp', '.png', '.jpg', '.jpeg'}

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

# 路径/I-O
def normalize_path(path):
    if not path:
        return path
    path = path.replace('\\', os.sep).replace('/', os.sep)
    return os.path.normpath(path)

def ensure_dir_exists(dir_path):
    try:
        dir_path = normalize_path(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"创建目录失败: {e}")
        return False

def safe_join(*paths):
    return normalize_path(os.path.join(*paths))

def imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
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
    try:
        path = normalize_path(path)
        parent_dir = os.path.dirname(path)
        if not ensure_dir_exists(parent_dir):
            return False
        ext = os.path.splitext(path)[1].lower() or ".tif"
        if not IS_WINDOWS or path.isascii():
            if ext in (".tif", ".tiff"):
                params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
                return cv2.imwrite(path, image, params)
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