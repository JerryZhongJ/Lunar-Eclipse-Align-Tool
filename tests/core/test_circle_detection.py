"""
圆检测功能单元测试
"""

import pytest
import cv2
import numpy as np
import os
from lunar_eclipse_align.core.circle_detection import (
    detect_circle,
    detect_circle_robust,
    standard_hough_detect,
    adaptive_hough_detect,
    contour_detect,
    padding_fallback_detect,
    final_detect,
    initial_process,
    build_detection_roi,
    BrightnessMode
)
from lunar_eclipse_align.core.utils import HoughParams
from lunar_eclipse_align.core.image import Image


@pytest.fixture
def test_setup():
    """共享的测试设置，准备图像和参数"""
    # 获取测试图像路径
    test_image_path = os.path.join(
        os.path.dirname(__file__), "..", "test_images", "basic", "DSC_3549.tif"
    )

    # 检查测试图像是否存在
    assert os.path.exists(test_image_path), f"测试图像不存在: {test_image_path}"

    # 读取图像
    cv_image = cv2.imread(test_image_path)
    assert cv_image is not None, f"无法读取测试图像: {test_image_path}"

    # 创建 Image 对象
    image = Image(bgr=cv_image)

    # 创建检测参数
    height, width = image.height, image.width
    min_radius = min(width, height) // 20
    max_radius = min(width, height) // 4

    params = HoughParams(
        minRadius=min_radius,
        maxRadius=max_radius,
        param1=50,
        param2=30
    )

    # 准备预处理后的数据
    gray, brightness_mode = initial_process(image, False)
    masked_gray = build_detection_roi(gray, params, None)

    return {
        'image': image,
        'params': params,
        'gray': gray,
        'masked_gray': masked_gray,
        'brightness_mode': brightness_mode
    }


def test_detect_circle_basic(test_setup):
    """基本的圆检测测试，使用真实月食图像"""
    image = test_setup['image']
    params = test_setup['params']

    # 调用 detect_circle 函数
    result = detect_circle(image, params)

    # 验证返回结果
    assert result is not None, "detect_circle 应该返回圆检测结果"

    # 检查圆心坐标是否在图像范围内
    height, width = image.height, image.width
    assert 0 <= result.x <= width, f"圆心 x 坐标 {result.x} 超出图像宽度 {width}"
    assert 0 <= result.y <= height, f"圆心 y 坐标 {result.y} 超出图像高度 {height}"

    # 检查半径是否为正数
    assert result.radius > 0, f"半径 {result.radius} 应该为正数"

    print(f"检测到的圆: 中心({result.x:.1f}, {result.y:.1f}), 半径{result.radius:.1f}")


def test_detect_circle_robust(test_setup):
    """测试稳健RANSAC圆检测"""
    gray = test_setup['gray']

    # 调用 detect_circle_robust 函数
    results = detect_circle_robust(gray)

    # 验证返回结果是列表
    assert isinstance(results, list), "detect_circle_robust 应该返回列表"

    # 如果检测到圆，验证其属性
    if results:
        circle = results[0]
        assert hasattr(circle, 'x'), "圆对象应该有 x 属性"
        assert hasattr(circle, 'y'), "圆对象应该有 y 属性"
        assert hasattr(circle, 'radius'), "圆对象应该有 radius 属性"
        assert circle.radius > 0, f"半径 {circle.radius} 应该为正数"
        print(f"RANSAC检测到的圆: 中心({circle.x:.1f}, {circle.y:.1f}), 半径{circle.radius:.1f}")
    else:
        print("RANSAC未检测到圆")


def test_standard_hough_detect(test_setup):
    """测试标准霍夫圆检测"""
    masked_gray = test_setup['masked_gray']
    params = test_setup['params']

    # 调用 standard_hough_detect 函数
    results = standard_hough_detect(masked_gray, params)

    # 验证返回结果是列表
    assert isinstance(results, list), "standard_hough_detect 应该返回列表"

    # 如果检测到圆，验证其属性
    if results:
        circle = results[0]
        assert hasattr(circle, 'x'), "圆对象应该有 x 属性"
        assert hasattr(circle, 'y'), "圆对象应该有 y 属性"
        assert hasattr(circle, 'radius'), "圆对象应该有 radius 属性"
        assert circle.radius > 0, f"半径 {circle.radius} 应该为正数"
        print(f"标准霍夫检测到的圆: 中心({circle.x:.1f}, {circle.y:.1f}), 半径{circle.radius:.1f}")
    else:
        print("标准霍夫未检测到圆")


def test_adaptive_hough_detect(test_setup):
    """测试自适应霍夫圆检测"""
    masked_gray = test_setup['masked_gray']
    params = test_setup['params']
    brightness_mode = test_setup['brightness_mode']

    # 调用 adaptive_hough_detect 函数
    results = adaptive_hough_detect(masked_gray, params, brightness_mode)

    # 验证返回结果是列表
    assert isinstance(results, list), "adaptive_hough_detect 应该返回列表"

    # 如果检测到圆，验证其属性
    if results:
        circle = results[0]
        assert hasattr(circle, 'x'), "圆对象应该有 x 属性"
        assert hasattr(circle, 'y'), "圆对象应该有 y 属性"
        assert hasattr(circle, 'radius'), "圆对象应该有 radius 属性"
        assert circle.radius > 0, f"半径 {circle.radius} 应该为正数"
        print(f"自适应霍夫检测到的圆: 中心({circle.x:.1f}, {circle.y:.1f}), 半径{circle.radius:.1f}")
    else:
        print("自适应霍夫未检测到圆")


def test_contour_detect(test_setup):
    """测试轮廓圆检测"""
    gray = test_setup['gray']
    params = test_setup['params']

    # 调用 contour_detect 函数
    results = contour_detect(gray, params)

    # 验证返回结果是列表
    assert isinstance(results, list), "contour_detect 应该返回列表"

    # 如果检测到圆，验证其属性
    if results:
        circle = results[0]
        assert hasattr(circle, 'x'), "圆对象应该有 x 属性"
        assert hasattr(circle, 'y'), "圆对象应该有 y 属性"
        assert hasattr(circle, 'radius'), "圆对象应该有 radius 属性"
        assert circle.radius > 0, f"半径 {circle.radius} 应该为正数"
        print(f"轮廓检测到的圆: 中心({circle.x:.1f}, {circle.y:.1f}), 半径{circle.radius:.1f}")
    else:
        print("轮廓检测未检测到圆")


def test_padding_fallback_detect(test_setup):
    """测试padding降级圆检测"""
    gray = test_setup['gray']
    params = test_setup['params']

    # 调用 padding_fallback_detect 函数
    results = padding_fallback_detect(gray, params)

    # 验证返回结果是列表
    assert isinstance(results, list), "padding_fallback_detect 应该返回列表"

    # 如果检测到圆，验证其属性
    if results:
        circle = results[0]
        assert hasattr(circle, 'x'), "圆对象应该有 x 属性"
        assert hasattr(circle, 'y'), "圆对象应该有 y 属性"
        assert hasattr(circle, 'radius'), "圆对象应该有 radius 属性"
        assert circle.radius > 0, f"半径 {circle.radius} 应该为正数"
        print(f"padding降级检测到的圆: 中心({circle.x:.1f}, {circle.y:.1f}), 半径{circle.radius:.1f}")
    else:
        print("padding降级检测未检测到圆")


def test_final_detect(test_setup):
    """测试最终圆检测"""
    gray = test_setup['gray']
    params = test_setup['params']

    # 调用 final_detect 函数
    results = final_detect(gray, params)

    # 验证返回结果是列表
    assert isinstance(results, list), "final_detect 应该返回列表"

    # 如果检测到圆，验证其属性
    if results:
        circle = results[0]
        assert hasattr(circle, 'x'), "圆对象应该有 x 属性"
        assert hasattr(circle, 'y'), "圆对象应该有 y 属性"
        assert hasattr(circle, 'radius'), "圆对象应该有 radius 属性"
        assert circle.radius > 0, f"半径 {circle.radius} 应该为正数"
        print(f"最终检测到的圆: 中心({circle.x:.1f}, {circle.y:.1f}), 半径{circle.radius:.1f}")
    else:
        print("最终检测未检测到圆")
