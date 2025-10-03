"""
圆检测参数化测试 - 使用配置文件
"""

import pytest
import yaml
from pathlib import Path
from lunar_eclipse_align.core.circle_detection import detect_circle
from lunar_eclipse_align.utils.data_types import HoughParams
from lunar_eclipse_align.utils.image import ImageFile


TEST_DATA_DIR = Path(__file__).parent / "test_data"


def load_images_config():
    """加载图像配置"""
    with open(TEST_DATA_DIR / "images.yaml") as f:
        return yaml.safe_load(f)["images"]


def load_params_config():
    """加载参数配置"""
    with open(TEST_DATA_DIR / "hough_params.yaml") as f:
        return yaml.safe_load(f)["param_sets"]


# ============ 统一的失败消息格式化函数 ============


def format_failure_message(
    img_name,
    img_data,
    param_name,
    param_data,
    params,
    expected,
    result=None,
    tolerance=None,
):
    """
    统一的失败消息格式化函数

    Args:
        result: 检测结果，None 表示未检测到圆
        tolerance: 容差，仅当 result 不为 None 时需要
    """
    # 头部和基本信息
    if result is None:
        title = "检测失败: 未检测到圆"
    else:
        title = "检测结果超出容差范围"

    msg = (
        f"\n"
        f"{'='*60}\n"
        f"{title}\n"
        f"{'='*60}\n"
        f"图像: {img_name} ({img_data['description']})\n"
        f"参数: {param_name} ({param_data['description']})\n"
        f"  minRadius={params.minRadius}, maxRadius={params.maxRadius}\n"
        f"  param1={params.param1}, param2={params.param2}\n"
    )

    # 如果有检测结果，添加结果和差异分析
    if result is not None and tolerance is not None:
        x_diff = abs(result.x - expected["x"])
        y_diff = abs(result.y - expected["y"])
        radius_diff = abs(result.radius - expected["radius"])

        msg += (
            f"\n"
            f"检测结果: 圆心({result.x:.2f}, {result.y:.2f}), 半径={result.radius:.2f}\n"
            f"预期结果: 圆心({expected['x']:.2f}, {expected['y']:.2f}), 半径={expected['radius']:.2f}\n"
            f"\n"
            f"差异分析:\n"
            f"  X 差异    = {x_diff:6.2f} (容差 < {tolerance['x']}){' ✓' if x_diff < tolerance['x'] else ' ✗'}\n"
            f"  Y 差异    = {y_diff:6.2f} (容差 < {tolerance['y']}){' ✓' if y_diff < tolerance['y'] else ' ✗'}\n"
            f"  半径差异  = {radius_diff:6.2f} (容差 < {tolerance['radius']}){' ✓' if radius_diff < tolerance['radius'] else ' ✗'}\n"
        )
    else:
        # 只有预期结果
        msg += f"预期结果: 圆心({expected['x']:.2f}, {expected['y']:.2f}), 半径={expected['radius']:.2f}\n"

    msg += f"{'='*60}"
    return msg


def check_result(result, expected, tolerance):
    """检查结果是否在容差范围内"""
    if result is None:
        return False, None, None, None

    x_diff = abs(result.x - expected["x"])
    y_diff = abs(result.y - expected["y"])
    radius_diff = abs(result.radius - expected["radius"])

    in_tolerance = (
        x_diff < tolerance["x"]
        and y_diff < tolerance["y"]
        and radius_diff < tolerance["radius"]
    )

    return in_tolerance, x_diff, y_diff, radius_diff


# ============ 测试1: 所有图片使用各自默认参数 ============
@pytest.mark.parametrize("img_name", load_images_config().keys())
def test_all_images(img_name):
    """测试所有图像，每个图像使用其默认参数"""
    images_config = load_images_config()
    params_config = load_params_config()

    img_data = images_config[img_name]
    param_name = img_data["default_params"]
    param_data = params_config[param_name]

    # 打印测试配置
    print(f"\n[测试配置] 图像={img_name}, 参数={param_name}")

    # 加载图像
    img_path = Path(img_data["path"])
    image = ImageFile(img_path).image
    assert image is not None, f"无法读取图像: {img_path}"

    # 创建参数
    params = HoughParams(**{k: v for k, v in param_data.items() if k != "description"})

    # 执行检测
    result = detect_circle(image, params)

    # 验证结果
    expected = img_data["expected_result"]
    tolerance = img_data["tolerance"]

    if result is None:
        pytest.fail(
            format_failure_message(
                img_name, img_data, param_name, param_data, params, expected
            )
        )

    in_tolerance, _, _, _ = check_result(result, expected, tolerance)

    assert in_tolerance, format_failure_message(
        img_name, img_data, param_name, param_data, params, expected, result, tolerance
    )


# ============ 测试2: 手动指定图像和参数 ============

# 在这里修改你想测试的图像和参数
MANUAL_TEST_IMAGE = "bright"  # 修改这里选择图像
MANUAL_TEST_PARAM = "default"  # 修改这里选择参数


def test_manually():
    """手动指定图像和参数进行测试 - 修改上面的常量来选择测试对象"""
    images_config = load_images_config()
    params_config = load_params_config()

    img_data = images_config[MANUAL_TEST_IMAGE]
    param_data = params_config[MANUAL_TEST_PARAM]

    # 打印测试配置
    print(f"\n[测试配置] 图像={MANUAL_TEST_IMAGE}, 参数={MANUAL_TEST_PARAM}")

    # 加载图像
    img_path = Path(img_data["path"])
    image = ImageFile(img_path).image
    assert image is not None, f"无法读取图像: {img_path}"

    # 创建参数
    params = HoughParams(**{k: v for k, v in param_data.items() if k != "description"})

    # 执行检测
    result = detect_circle(image, params)

    # 验证结果
    expected = img_data["expected_result"]
    tolerance = img_data["tolerance"]

    if result is None:
        pytest.fail(
            format_failure_message(
                MANUAL_TEST_IMAGE,
                img_data,
                MANUAL_TEST_PARAM,
                param_data,
                params,
                expected,
            )
        )

    in_tolerance, _, _, _ = check_result(result, expected, tolerance)

    assert in_tolerance, format_failure_message(
        MANUAL_TEST_IMAGE,
        img_data,
        MANUAL_TEST_PARAM,
        param_data,
        params,
        expected,
        result,
        tolerance,
    )


# ============ 测试3: 探索特定图像的所有参数 ============

EXPLORE_IMAGE = "bright"  # 修改这里选择要探索的图像


@pytest.mark.parametrize("param_name", load_params_config().keys())
def test_all_params_for_image(param_name):
    """对指定图像测试所有参数配置 - 用于参数调优"""
    images_config = load_images_config()
    params_config = load_params_config()

    assert (
        EXPLORE_IMAGE in images_config
    ), f"图像 '{EXPLORE_IMAGE}' 不存在。可用的图像: {list(images_config.keys())}"

    img_data = images_config[EXPLORE_IMAGE]
    param_data = params_config[param_name]

    # 打印测试配置
    print(f"\n[探索] 图像={EXPLORE_IMAGE}, 参数={param_name}")

    # 加载图像
    img_path = Path(img_data["path"])
    image = ImageFile(img_path).image
    assert image is not None, f"无法读取图像: {img_path}"

    # 创建参数
    params = HoughParams(**{k: v for k, v in param_data.items() if k != "description"})

    # 执行检测
    result = detect_circle(image, params)

    # 验证结果
    expected = img_data["expected_result"]
    tolerance = img_data["tolerance"]

    if result is None:
        pytest.fail(
            format_failure_message(
                EXPLORE_IMAGE, img_data, param_name, param_data, params, expected
            )
        )

    in_tolerance, _, _, _ = check_result(result, expected, tolerance)

    assert in_tolerance, format_failure_message(
        EXPLORE_IMAGE,
        img_data,
        param_name,
        param_data,
        params,
        expected,
        result,
        tolerance,
    )
