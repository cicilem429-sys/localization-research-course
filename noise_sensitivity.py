
import numpy as np

from localization_tools import estimate_position_multistart
from geometry_consistency import detect_outliers
from hybrid_localization import hybrid_localization

anchors = np.array([
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10],
    [5, 15]
], dtype=float)

true_position = np.array([4.0, 6.0])
true_distances = np.linalg.norm(
    anchors - true_position, axis=1
)

noise_levels = [0.1, 0.2, 0.5, 1.0]
trials = 100
rng = np.random.default_rng(42)

for sigma in noise_levels:
    errors = {
        "OLS": [],
        "Huber": [],
        "Geometry": [],
        "Hybrid": []
    }

    correct = 0

    for _ in range(trials):
        distances = true_distances + rng.normal(
            0, sigma, size=len(anchors)
        )

        # 固定第4个节点存在异常测距
        distances[3] += 8.0

        # 距离不能为负，避免高噪声下输入非法
        distances = np.maximum(distances, 0.0)

        ols = estimate_position_multistart(
            anchors, distances, method="linear"
        )

        huber = estimate_position_multistart(
            anchors, distances,
            method="huber", delta=0.3
        )

        geometry = detect_outliers(
            anchors, distances, threshold=1.0
        )

        hybrid = hybrid_localization(
            anchors, distances,
            threshold=1.0, delta=0.3
        )

        estimates = {
            "OLS": ols["position"],
            "Huber": huber["position"],
            "Geometry": geometry["estimated_position"],
            "Hybrid": hybrid["position"]
        }

        for name, position in estimates.items():
            error = np.linalg.norm(
                np.asarray(position) - true_position
            )
            errors[name].append(error)

        if geometry["suspected_outliers"] == [3]:
            correct += 1

    print(f"\n===== 噪声标准差 sigma={sigma} =====")

    for name, values in errors.items():
        print(
            f"{name}: "
            f"平均误差={np.mean(values):.4f}, "
            f"最大误差={np.max(values):.4f}"
        )

    print(
        f"异常节点完全识别率：{correct / trials:.2%}"
    )
