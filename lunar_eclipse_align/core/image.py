import logging
from pathlib import Path
from typing import Literal
from PIL import Image as PILImage

import cv2
import numpy as np
from numpy.typing import NDArray

import gc

from lunar_eclipse_align.core.utils import SUPPORTED_EXTS


class Image:
    _rgb: NDArray
    _bgr: NDArray | None = None
    exif: PILImage.Exif | None
    icc: bytes | None
    _normalized_gray: NDArray | None = None
    _width: int
    _height: int

    def __init__(
        self,
        *,
        rgb: NDArray | None = None,
        bgr: NDArray | None = None,
        exif: PILImage.Exif | None = None,
        icc: bytes | None = None,
    ):

        if rgb is not None:
            self._rgb = rgb
        elif bgr is not None:
            self._rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        else:
            raise ValueError("必须提供 rgb 或 bgr 数据")
        self._height, self._width = self.rgb.shape[:2]
        self.exif = exif
        self.icc = icc

    @staticmethod
    def from_file(file_path: Path) -> "Image | None":
        """从文件加载图像"""
        try:
            pil_image = PILImage.open(file_path)
        except Exception as e:
            logging.error(f"无法加载图像 {file_path}: {e}")
            return None
        exif: PILImage.Exif = pil_image.getexif()
        icc: bytes | None = pil_image.info.get("icc_profile", None)
        rgb = np.array(pil_image)
        return Image(rgb=rgb, exif=exif, icc=icc)

    def save(self, file_path: Path) -> bool:
        """保存图像到文件"""
        save_kwargs = {}
        suffix = file_path.suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            save_kwargs["quality"] = 95
        elif suffix in [".tif", ".tiff"]:
            save_kwargs["compression"] = "tiff_deflate"
        try:
            pil_image = PILImage.fromarray(self.rgb)
            pil_image.save(
                file_path, exif=self.exif, icc_profile=self.icc, **save_kwargs
            )
        except Exception as e:
            logging.error(f"无法保存图像到 {file_path}: {e}")
            return False
        return True

    @property
    def rgb(self) -> NDArray:
        """获取 RGB 格式的图像数据"""
        return self._rgb

    @property
    def bgr(self) -> NDArray:
        """获取 BGR 格式的图像数据"""
        if self._bgr is None:
            self._bgr = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        return self._bgr

    @property
    def normalized_gray(self) -> NDArray:
        """获取灰度图像数据"""
        if self._normalized_gray is None:
            gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
            self._normalized_gray = cv2.normalize(
                gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U  # type: ignore
            )
            assert self._normalized_gray is not None
        return self._normalized_gray

    @property
    def width(self) -> int:
        """获取图像宽度"""
        return self._width

    @property
    def height(self) -> int:
        """获取图像高度"""
        return self._height

    @property
    def widthXheight(self) -> tuple[int, int]:
        """获取图像的高度和宽度"""
        return self._width, self._height


class ImageFile:
    """
    图像文件的包装类，支持按需加载和保存
    通过 mode 参数控制读写模式：
    - "r": 只读模式，按需加载图像
    - "w": 写入模式，允许设置和保存图像
    """

    _image: Image | None = None
    _path: Path
    _mode: Literal["r"] | Literal["w"]

    def __init__(self, file_path: Path, mode: Literal["r"] | Literal["w"] = "r"):
        self._path = file_path
        self._mode = mode

    @staticmethod
    def load(dir: Path) -> "dict[Path, ImageFile]":
        """创建 ImageFile 实例"""
        image_files = []
        for ext in SUPPORTED_EXTS:
            image_files.extend(dir.glob(ext))
        return {fp: ImageFile(fp) for fp in sorted(image_files)}

    @property
    def image(self) -> Image | None:
        """获取图像对象，按需加载"""
        if self._mode == "r" and self._image is None:
            self._image = Image.from_file(self._path)
        return self._image

    @image.setter
    def image(self, image: Image):
        """设置图像对象"""
        if self._mode == "w":
            self._image = image
        else:
            raise ValueError("只能在写模式下设置图像")

    def __del__(self):
        """析构时释放图像数据"""
        if self._image is not None:
            self._image = None
            gc.collect()

    @property
    def path(self) -> Path:
        return self._path

    def save(self) -> bool:
        """保存图像到文件"""
        if self._mode != "w":
            raise ValueError("只能在写模式下保存图像")
        if self._image is None:
            raise ValueError("没有图像数据可保存")
        return self._image.save(self._path)


if __name__ == "__main__":
    # 测试代码
    image = Image.from_file(Path("tests/test_images/basic/DSC_3551.tif"))
