
import numpy as np

from localization_tools import estimate_position_multistart
from fast_geometry import fast_geometry_localization


def cep90(errors):
    """返回定位误差的第90百分位数。"""
    return float(np.percentile(errors, 90))


rng = np.random.default_rng(2028)

# 十个固定观测节点，分布在一个二维区域
anchors = np.array([
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10],
    [5, 15],
    [15, 5],
    [-5, 5],
    [5, -5],
    [15, 15],
    [-5, 15]
], dtype=float)

noise_levels = [0.2, 0.5, 1.0]
trials = 50

for sigma in noise_levels:

    errors = {
        "OLS": [],
        "Huber": [],
        "FastGeometry": []
    }

    correct_detections = 0

    for _ in range(trials):

        # 每次实验随机改变目标位置
        true_position = rng.uniform(
            low=[2, 2],
            high=[8, 8]
        )

        true_distances = np.linalg.norm(
            anchors - true_position,
            axis=1
        )

        # 加入正常高斯测距噪声
        distances = true_distances + rng.normal(
            0, sigma, size=len(anchors)
        )

        # 随机选择一个异常节点
        outlier_index = int(
            rng.integers(0, len(anchors))
        )

        distances[outlier_index] += 8.0

        distances = np.maximum(distances, 0.0)

        ols = estimate_position_multistart(
            anchors,
            distances,
            method="linear"
        )

        huber = estimate_position_multistart(
            anchors,
            distances,
            method="huber",
            delta=0.3
        )

        geometry = fast_geometry_localization(
            anchors,
            distances,
            threshold=1.0
        )

        estimates = {
            "OLS": ols["position"],
            "Huber": huber["position"],
            "FastGeometry": geometry["position"]
        }

        for name, position in estimates.items():
            error = np.linalg.norm(
                np.asarray(position) - true_position
            )

            errors[name].append(float(error))

        if geometry["suspected_outliers"] == [
            outlier_index
        ]:
            correct_detections += 1

    print(f"\n===== 十节点 sigma={sigma} =====")

    for name, values in errors.items():

        print(f"\n方法：{name}")

        print(
            "平均定位误差：",
            round(np.mean(values), 4)
        )

        print(
            "CEP90：",
            round(cep90(values), 4)
        )

        print(
            "最大定位误差：",
            round(np.max(values), 4)
        )

    print(
        "\n异常节点完全识别率：",
        f"{correct_detections / trials:.1%}"
    )
