# utils_common.py
from dataclasses import dataclass
import math
from pathlib import Path
import os, platform, gc
from typing import Any, Generator, Generic, NamedTuple, Type, TypeVar, overload
import numpy as np
import cv2


from numpy.typing import NDArray

# ----------------- 系统/常量 -----------------
SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_MACOS = SYSTEM == "Darwin"
IS_LINUX = SYSTEM == "Linux"
SUPPORTED_EXTS = {"*.tif", "*.tiff", "*.bmp", "*.png", "*.jpg", "*.jpeg"}

# 内存管理
MAX_IMAGES_IN_MEMORY = 10
MEMORY_THRESHOLD_MB = 500
MAX_SCAN_COUNT = 10
MAX_SIDE = 1600  # 图像最长边缩略尺寸
MAX_REFINE_DELTA_PX = 6.0
MIN_MEAN_ZNCC = 0.55
MIN_INLIERS = 6


def get_memory_usage_mb():
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


class Hough(NamedTuple):
    minRadius: int
    maxRadius: int
    param1: int
    param2: int


# def imread_unicode(
#     path: Path, flags=cv2.IMREAD_UNCHANGED
# ) -> np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]] | None:
#     try:
#         data = np.fromfile(path, dtype=np.uint8)
#         return cv2.imdecode(data, flags)
#     except Exception as e:
#         print(f"图像读取失败 {path}: {e}")
#         return None


# def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
#     try:
#         path.parent.mkdir(parents=True, exist_ok=True)
#         ext = path.suffix.lower() or ".tiff"

#         if ext in (".tif", ".tiff"):
#             params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
#         else:
#             params = []
#         ok, buf = cv2.imencode(ext, image, params)
#         if not ok:
#             raise Exception("图像编码失败")
#         buf.tofile(path)
#         return True
#     except Exception as e:
#         print(f"图像保存失败 {path}: {e}")
#         return False


# def imwrite_with_exif(src_path: Path, dst_path: Path, img_bgr: np.ndarray) -> bool:
#     """
#     优先保留 src_path 的 EXIF/ICC，用 Pillow 写出 JPEG/TIFF。
#     失败或不支持时回退到 imwrite_unicode。
#     参数:
#         src_path: 原始读取图像的文件路径(用于提取EXIF/ICC)
#         dst_path: 目标输出路径
#         img_bgr:  OpenCV(BGR) 图像
#     返回: bool 是否写出成功
#     """

#     dst_path.parent.mkdir(parents=True, exist_ok=True)

#     ext = dst_path.suffix.lower() or ".tiff"
#     # 仅在 JPEG/TIFF 尝试保EXIF，其余格式走原逻辑
#     if ext not in (".jpg", ".jpeg", ".tif", ".tiff"):
#         return imwrite_unicode(dst_path, img_bgr)

#     # BGR -> RGB
#     try:
#         rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     except Exception:
#         # 如果颜色转换失败，仍尝试直接构建
#         rgb = img_bgr
#     im = Image.fromarray(rgb)

#     exif_bytes = None
#     icc = None
#     # 读取源图的 EXIF 与 ICC

#     with Image.open(src_path) as src_im:
#         exif_bytes = src_im.info.get("exif", None)
#         icc = src_im.info.get("icc_profile", None)
#         # 将 Orientation 归一到 1，避免查看器再次旋转
#         if exif_bytes:

#             exif_dict = piexif.load(exif_bytes)
#             exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
#             exif_bytes = piexif.dump(exif_dict)

#     save_kwargs = {}
#     if exif_bytes is not None:
#         save_kwargs["exif"] = exif_bytes
#     if icc is not None:
#         save_kwargs["icc_profile"] = icc

#     if ext in (".jpg", ".jpeg"):
#         # 设一个较高质量，保持文件体积与质量平衡
#         save_kwargs.setdefault("quality", 95)
#         im.save(dst_path, **save_kwargs)
#     else:  # TIFF
#         # 无损/轻压缩
#         save_kwargs.setdefault("compression", "tiff_deflate")
#         im.save(dst_path, **save_kwargs)
#     return True


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
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)


dataType = TypeVar("dataType", int, float)


@dataclass(frozen=True)
class Position(Generic[dataType]):
    x: dataType
    y: dataType

    def __add__(self, delta: "Vector[dataType]") -> "Position[dataType]":
        return Position(self.x + delta.x, self.y + delta.y)

    @overload
    def __sub__(self, other: "Position[dataType]") -> "Vector[dataType]": ...
    @overload
    def __sub__(self, other: "Vector[dataType]") -> "Position[dataType]": ...
    def __sub__(
        self, other: "Position[dataType] | Vector[dataType]"
    ) -> "Vector[dataType] | Position[dataType]":
        if isinstance(other, Vector):
            return Position(self.x - other.x, self.y - other.y)
        else:
            return Vector(self.x - other.x, self.y - other.y)

    dataType2 = TypeVar("dataType2", int, float)

    def as_type(self, dType: Type[dataType2]) -> "Position[dataType2]":
        return Position(dType(self.x), dType(self.y))


class PositionArray:
    def __init__(self, arr: NDArray, safe: bool = True):
        assert arr.ndim == 2 and arr.shape[1] == 2
        if safe:
            self._arr = arr.astype(np.float64, copy=True)
        else:
            self._arr = arr

    def __iter__(self) -> Generator[Position, Any, None]:
        for x, y in self._arr:
            yield Position(x, y)

    def __add__(self, vector: "Vector[float] | VectorArray") -> "PositionArray":
        if isinstance(vector, VectorArray):
            return PositionArray(self._arr + vector._arr, safe=False)
        else:
            return PositionArray(
                self._arr + np.array([[vector.x, vector.y]]), safe=False
            )

    def __sub__(self, other: "Position[float] | PositionArray") -> "VectorArray":
        if isinstance(other, Position):
            return VectorArray(self._arr - np.array([[other.x, other.y]]), safe=False)
        else:
            return VectorArray(self._arr - other._arr, safe=False)

    def __len__(self) -> int:
        return self._arr.shape[0]

    @property
    def x(self) -> NDArray[np.float64]:
        return self._arr[:, 0]

    @property
    def y(self) -> NDArray[np.float64]:
        return self._arr[:, 1]

    def filter(self, mask: NDArray[np.bool_]) -> "PositionArray":
        return PositionArray(self._arr[mask], safe=False)


@dataclass(frozen=True)
class Vector(Generic[dataType]):
    x: dataType
    y: dataType

    dataType2 = TypeVar("dataType2", int, float)

    @staticmethod
    def from_ndarray(arr: NDArray, dataType: Type[dataType2]) -> "Vector[dataType2]":
        assert arr.ndim == 1 and arr.shape[0] == 2
        return Vector(dataType(arr[0]), dataType(arr[1]))

    def as_type(self, dType: Type[dataType2]) -> "Vector[dataType2]":
        return Vector(dType(self.x), dType(self.y))

    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    def normalize(self) -> "Vector[float]":
        n = self.norm() + 1e-6
        return Vector(self.x / n, self.y / n)

    def __add__(self, other: "Vector[dataType]") -> "Vector[dataType]":
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector[dataType]") -> "Vector[dataType]":
        return Vector(self.x - other.x, self.y - other.y)

    @overload
    def __mul__(self, other: float) -> "Vector[float]": ...
    @overload
    def __mul__(self, other: "Vector[dataType]") -> dataType: ...

    def __mul__(self, other: "float | Vector[dataType]") -> "float | Vector[float]":
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y
        else:
            return Vector(self.x * other, self.y * other)

    def __div__(self, other: float) -> "Vector":
        return Vector(self.x / other, self.y / other)


class VectorArray:
    """
    为了节省性能，VectorArray直接使用
    """

    def __init__(self, arr: NDArray, safe: bool = True):
        assert arr.ndim == 2 and arr.shape[1] == 2
        if safe:
            self._arr = arr.astype(np.float64, copy=True)
        else:
            self._arr = arr

    def norms(self) -> NDArray[np.float64]:
        return np.hypot(self._arr[:, 0], self._arr[:, 1])

    def normalize(self) -> "VectorArray":
        norms = self.norms() + 1e-6
        return VectorArray(self._arr / norms[:, np.newaxis], safe=False)

    @overload
    def __mul__(self, other: float | NDArray[np.floating]) -> "VectorArray": ...
    @overload
    def __mul__(self, other: "VectorArray | Vector[float]") -> NDArray[np.float64]: ...
    def __mul__(
        self, other: "float | NDArray[np.floating] | VectorArray | Vector[float]"
    ) -> "NDArray[np.float64] | VectorArray":
        if isinstance(other, float):
            return VectorArray(self._arr * other, safe=False)
        elif isinstance(other, np.ndarray):
            assert other.ndim == 1 and other.shape[0] == self._arr.shape[0]
            return VectorArray(self._arr * other[:, np.newaxis], safe=False)

        elif isinstance(other, Vector):
            return self._arr[:, 0] * other.x + self._arr[:, 1] * other.y
        elif isinstance(other, VectorArray):
            return (
                self._arr[:, 0] * other._arr[:, 0] + self._arr[:, 1] * other._arr[:, 1]
            )
        raise TypeError("Unsupported type for multiplication")

    @overload
    def __add__(self, other: "VectorArray | Vector[float]") -> "VectorArray": ...
    @overload
    def __add__(self, other: "PositionArray | Position[float]") -> "PositionArray": ...
    def __add__(
        self, other: "VectorArray | Vector[float] | PositionArray | Position[float]"
    ) -> "VectorArray | PositionArray":
        if isinstance(other, Vector):
            return VectorArray(self._arr + np.array([[other.x, other.y]]), safe=False)
        elif isinstance(other, VectorArray):
            return VectorArray(self._arr + other._arr, safe=False)
        elif isinstance(other, Position):
            return PositionArray(self._arr + np.array([[other.x, other.y]]), safe=False)
        elif isinstance(other, PositionArray):
            return PositionArray(self._arr + other._arr, safe=False)
        raise TypeError("Unsupported type for addition")

    def __sub__(self, other: "VectorArray | Vector[float]") -> "VectorArray":
        if isinstance(other, Vector):
            return VectorArray(self._arr - np.array([[other.x, other.y]]), safe=False)
        else:
            return VectorArray(self._arr - other._arr, safe=False)

    def __div__(self, other: float) -> "VectorArray":
        return VectorArray(self._arr / other, safe=False)

    def __iter__(self) -> Generator[Vector[float], Any, None]:
        for x, y in self._arr:
            yield Vector(x, y)

    def __len__(self) -> int:
        return self._arr.shape[0]

    @property
    def x(self) -> NDArray[np.float64]:
        return self._arr[:, 0]

    @property
    def y(self) -> NDArray[np.float64]:
        return self._arr[:, 1]

    def filter(self, mask: NDArray[np.bool_]) -> "VectorArray":
        return VectorArray(self._arr[mask], safe=False)


@dataclass(frozen=True)
class Circle(Position[float]):
    radius: float

    @staticmethod
    def from_ndarray(arr: NDArray) -> "Circle":
        assert arr.ndim == 1 and arr.shape[0] == 3
        return Circle(float(arr[0]), float(arr[1]), float(arr[2]))

    def shift(self, delta: "Vector[float]") -> "Circle":
        return Circle(self.x + delta.x, self.y + delta.y, self.radius)

    def scale(self, factor: float) -> "Circle":
        return Circle(self.x, self.y, self.radius * factor)


@dataclass(frozen=True)
class ROI(Position[int]):
    w: int
    h: int
    score: float = 0.0  # 可选的置信度评分

    @property
    def size(self) -> Vector[int]:
        return Vector(self.w, self.h)

    @property
    def center(self) -> Position[float]:
        return Position(self.x + self.w / 2.0, self.y + self.h / 2.0)


class DetectionResult(NamedTuple):
    circle: Circle
    quality: float
