# utils_common.py
from abc import ABC
from dataclasses import dataclass
from functools import lru_cache
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


class HoughParams(NamedTuple):
    minRadius: int
    maxRadius: int
    param1: int
    param2: int
    method: int = cv2.HOUGH_GRADIENT
    dp: float = 1.2


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


CordType = TypeVar("CordType", int, float)


@dataclass(frozen=True)
class Positioned(ABC, Generic[CordType]):
    x: CordType
    y: CordType


@dataclass(frozen=True)
class Point(Positioned[CordType]):

    def __add__(self, delta: "Vector[CordType]") -> "Point[CordType]":
        return Point(self.x + delta.x, self.y + delta.y)

    @overload
    def __sub__(self, other: "Point[CordType]") -> "Vector[CordType]": ...
    @overload
    def __sub__(self, other: "Vector[CordType]") -> "Point[CordType]": ...
    def __sub__(
        self, other: "Point[CordType] | Vector[CordType]"
    ) -> "Vector[CordType] | Point[CordType]":
        if isinstance(other, Vector):
            return Point(self.x - other.x, self.y - other.y)
        else:
            return Vector(self.x - other.x, self.y - other.y)

    dataType2 = TypeVar("dataType2", int, float)

    def as_type(self, dType: Type[dataType2]) -> "Point[dataType2]":
        return Point(dType(self.x), dType(self.y))


ORIGIN = Point(0.0, 0.0)
ORIGIN_I = Point(0, 0)


@dataclass(frozen=True)
class Boundary(Generic[CordType]):
    min_x: CordType
    max_x: CordType
    min_y: CordType
    max_y: CordType

    def bound(self, point: "Point[CordType]") -> "BoundedPoint[CordType]":
        x = point.x
        y = point.y
        if x < self.min_x:
            x = self.min_x
        if x > self.max_x:
            x = self.max_x
        if y < self.min_y:
            y = self.min_y
        if y > self.max_y:
            y = self.max_y
        return BoundedPoint(x, y, boundary=self)


@dataclass(frozen=True)
class BoundedPoint(Positioned[CordType]):
    boundary: Boundary[CordType]

    def __add__(self, delta: "Vector[CordType]") -> "BoundedPoint[CordType]":
        p = Point(self.x + delta.x, self.y + delta.y)
        return self.boundary.bound(p)

    @overload
    def __sub__(self, other: "BoundedPoint[CordType]") -> "Vector[CordType]": ...
    @overload
    def __sub__(self, other: "Vector[CordType]") -> "BoundedPoint[CordType]": ...
    def __sub__(
        self, other: "BoundedPoint[CordType] | Vector[CordType]"
    ) -> "Vector[CordType] | BoundedPoint[CordType]":
        if isinstance(other, Vector):
            p = Point(self.x - other.x, self.y - other.y)
            return self.boundary.bound(p)
        else:
            if self.boundary != other.boundary:
                raise Exception(
                    "Cannot subtract BoundedPoints with different boundaries"
                )
            return Vector(self.x - other.x, self.y - other.y)


class PointArray:
    def __init__(self, arr: NDArray, safe: bool = True):
        assert arr.ndim == 2 and arr.shape[1] == 2
        if safe:
            self._arr = arr.astype(np.float64, copy=True)
        else:
            self._arr = arr

    def __iter__(self) -> Generator[Point, Any, None]:
        for x, y in self._arr:
            yield Point(x, y)

    def __add__(self, vector: "Vector[float] | VectorArray") -> "PointArray":
        if isinstance(vector, VectorArray):
            return PointArray(self._arr + vector._arr, safe=False)
        else:
            return PointArray(self._arr + np.array([[vector.x, vector.y]]), safe=False)

    def __sub__(self, other: "Point[float] | PointArray") -> "VectorArray":
        if isinstance(other, Point):
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

    def filter(self, mask: NDArray[np.bool_]) -> "PointArray":
        return PointArray(self._arr[mask], safe=False)


@dataclass(frozen=True)
class Vector(Positioned[CordType]):
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

    def __add__(self, other: "Vector[CordType]") -> "Vector[CordType]":
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector[CordType]") -> "Vector[CordType]":
        return Vector(self.x - other.x, self.y - other.y)

    @overload
    def __mul__(self, other: float) -> "Vector[float]": ...
    @overload
    def __mul__(self, other: "Vector[CordType]") -> CordType: ...

    def __mul__(self, other: "float | Vector[CordType]") -> "float | Vector[float]":
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y
        else:
            return Vector(self.x * other, self.y * other)

    def __div__(self, other: float) -> "Vector":
        return Vector(self.x / other, self.y / other)


ZERO = Vector(0.0, 0.0)
ZERO_I = Vector(0, 0)


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
    def __add__(self, other: "PointArray | Point[float]") -> "PointArray": ...
    def __add__(
        self, other: "VectorArray | Vector[float] | PointArray | Point[float]"
    ) -> "VectorArray | PointArray":
        if isinstance(other, Vector):
            return VectorArray(self._arr + np.array([[other.x, other.y]]), safe=False)
        elif isinstance(other, VectorArray):
            return VectorArray(self._arr + other._arr, safe=False)
        elif isinstance(other, Point):
            return PointArray(self._arr + np.array([[other.x, other.y]]), safe=False)
        elif isinstance(other, PointArray):
            return PointArray(self._arr + other._arr, safe=False)
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
class Circle(Positioned[float]):
    radius: float

    @staticmethod
    def from_ndarray(arr: NDArray) -> "Circle":
        assert arr.ndim == 1 and arr.shape[0] == 3
        return Circle(float(arr[0]), float(arr[1]), float(arr[2]))

    def scale(self, factor: float) -> "Circle":
        return Circle(self.x * factor, self.y * factor, self.radius * factor)

    def shift(self, v: Vector[float]) -> "Circle":
        return Circle(self.x + v.x, self.y + v.y, self.radius)

    @property
    def center(self) -> Point[float]:
        return Point(self.x, self.y)


@dataclass(frozen=True)
class ROI(Positioned[int]):
    w: int
    h: int
    score: float = 0.0  # 可选的置信度评分

    @property
    def size(self) -> Vector[int]:
        return Vector(self.w, self.h)

    @property
    def center(self) -> Point[float]:
        return Point(self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def position(self) -> Point[int]:
        return Point(self.x, self.y)


class DetectionResult(NamedTuple):
    circle: Circle
    quality: float


@lru_cache(maxsize=32)
def soft_disk_mask(
    height: int,
    width: int,
    circle: Circle,
    inner: float,
    outer: float,
) -> NDArray[np.float32]:

    dist = distance_map(width, height, circle)
    m = np.zeros((height, width), np.float32)
    r_in = circle.radius * inner
    r_out = circle.radius * outer

    band = ring_mask(width, height, circle, inner=inner, outer=outer)

    t = (dist[band] - r_in) / ((r_out - r_in) + 1e-6)
    m[band] = 0.5 * (1 + np.cos(np.pi * t))
    m[dist <= r_in] = 1.0
    return m


# TODO: a better way to optimize it is to build a class Detection to wrap these stuffs
# and cache all the intermediate ingredients inside an object
# they will be dropped as the detection ends
@lru_cache(maxsize=32)
def distance_map(width: int, height: int, circle: Circle) -> NDArray[np.float32]:
    Y, X = np.ogrid[:height, :width]
    dist = np.sqrt((X - circle.x) ** 2 + (Y - circle.y) ** 2)
    return dist


@lru_cache(maxsize=32)
def ring_mask(
    width: int, height: int, circle: Circle, inner: float, outer: float
) -> NDArray[np.bool]:
    dist = distance_map(width, height, circle)
    return (dist >= circle.radius * inner) & (dist <= circle.radius * outer)


def touches_border(width: int, height: int, circle: Circle, margin: int = 5) -> bool:
    return (
        (circle.x - circle.radius < margin)
        or (circle.y - circle.radius < margin)
        or (circle.x + circle.radius > width - margin)
        or (circle.y + circle.radius > height - margin)
    )


NUMBER = TypeVar("NUMBER", int, float)


def clip(value: NUMBER, min_value: NUMBER, max_value: NUMBER) -> NUMBER:
    return max(min_value, min(value, max_value))
