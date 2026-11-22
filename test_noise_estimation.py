
import numpy as np

from noise_estimation import estimate_noise

anchors = np.array([
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10],
    [5, 15]
], dtype=float)

true_position = np.array([4.0, 6.0])

true_distances = np.linalg.norm(
    anchors - true_position,
    axis=1
)

noise_levels = [0.1, 0.2, 0.5, 1.0]

trials = 100

# 固定随机种子，保证实验可复现
rng = np.random.default_rng(42)

for sigma in noise_levels:

    sigma_estimates = []
    correct_detections = 0

    for _ in range(trials):

        distances = true_distances + rng.normal(
            0, sigma, size=len(anchors)
        )

        # 第4个节点加入异常测距
        distances[3] += 8.0

        result = estimate_noise(
            anchors,
            distances,
            threshold=1.0
        )

        sigma_estimates.append(
            result["sigma_hat"]
        )

        if result["suspected_outliers"] == [3]:
            correct_detections += 1

    sigma_estimates = np.array(sigma_estimates)

    print(f"\n===== 真实 sigma = {sigma} =====")

    print(
        "平均估计噪声：",
        round(np.mean(sigma_estimates), 4)
    )

    print(
        "估计噪声标准差：",
        round(np.std(sigma_estimates), 4)
    )

    print(
        "估计噪声中位数：",
        round(np.median(sigma_estimates), 4)
    )

    print(
        "最小估计值：",
        round(np.min(sigma_estimates), 4)
    )

    print(
        "最大估计值：",
        round(np.max(sigma_estimates), 4)
    )

    print(
        "异常节点完全识别率：",
        f"{correct_detections / trials:.1%}"
    )
