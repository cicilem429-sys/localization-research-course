
import numpy as np

from geometry_consistency import detect_outliers
from adaptive_localization import adaptive_localization

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
rng = np.random.default_rng(2027)

for sigma in noise_levels:

    fixed_errors = []
    adaptive_errors = []

    fixed_correct = 0
    adaptive_correct = 0
    thresholds = []

    for _ in range(trials):

        distances = true_distances + rng.normal(
            0, sigma, size=len(anchors)
        )
        distances[3] += 8.0

        # 固定阈值方法
        fixed = detect_outliers(
            anchors, distances, threshold=1.0
        )

        # 自适应阈值方法
        adaptive = adaptive_localization(
            anchors, distances
        )

        fixed_position = np.asarray(
            fixed["estimated_position"]
        )
        adaptive_position = np.asarray(
            adaptive["position"]
        )

        fixed_errors.append(
            np.linalg.norm(
                fixed_position - true_position
            )
        )

        adaptive_errors.append(
            np.linalg.norm(
                adaptive_position - true_position
            )
        )

        thresholds.append(adaptive["threshold"])

        if fixed["suspected_outliers"] == [3]:
            fixed_correct += 1

        if adaptive["suspected_outliers"] == [3]:
            adaptive_correct += 1

    print(f"\n===== 真实 sigma = {sigma} =====")

    print(
        "固定阈值平均定位误差：",
        round(np.mean(fixed_errors), 4)
    )
    print(
        "自适应平均定位误差：",
        round(np.mean(adaptive_errors), 4)
    )
    print(
        "固定阈值异常识别率：",
        f"{fixed_correct / trials:.1%}"
    )
    print(
        "自适应异常识别率：",
        f"{adaptive_correct / trials:.1%}"
    )
    print(
        "平均自适应阈值：",
        round(np.mean(thresholds), 4)
    )
    print(
        "自适应最大定位误差：",
        round(np.max(adaptive_errors), 4)
    )
