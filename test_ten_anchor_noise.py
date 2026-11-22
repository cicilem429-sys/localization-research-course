
import numpy as np

from localization_tools import estimate_position_multistart


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

trials = 100
rng = np.random.default_rng(2031)


def estimate_noise_mad(anchors, distances):

    # 使用 Huber 获得初始位置
    result = estimate_position_multistart(
        anchors,
        distances,
        method="huber",
        delta=0.3
    )

    position = np.asarray(
        result["position"],
        dtype=float
    )

    # 计算有符号测距残差
    predicted = np.linalg.norm(
        anchors - position,
        axis=1
    )

    residuals = predicted - distances

    # MAD 对单个极端异常观测相对稳健
    median = np.median(residuals)

    mad = np.median(
        np.abs(residuals - median)
    )

    sigma_hat = 1.4826 * mad

    return float(sigma_hat)


for sigma in noise_levels:

    estimates = []

    for _ in range(trials):

        # 每次随机改变真实位置
        true_position = rng.uniform(
            low=[2, 2],
            high=[8, 8]
        )

        true_distances = np.linalg.norm(
            anchors - true_position,
            axis=1
        )

        distances = true_distances + rng.normal(
            0, sigma,
            size=len(anchors)
        )

        # 随机选择一个异常节点
        outlier_index = int(
            rng.integers(0, len(anchors))
        )

        distances[outlier_index] += 8.0
        distances = np.maximum(distances, 0.0)

        estimate = estimate_noise_mad(
            anchors,
            distances
        )

        estimates.append(estimate)

    print(f"\n===== 真实 sigma={sigma} =====")

    print(
        "平均估计噪声：",
        round(np.mean(estimates), 4)
    )

    print(
        "估计噪声中位数：",
        round(np.median(estimates), 4)
    )

    print(
        "估计值标准差：",
        round(np.std(estimates), 4)
    )

    print(
        "平均绝对误差：",
        round(
            np.mean(
                np.abs(
                    np.asarray(estimates) - sigma
                )
            ),
            4
        )
    )
