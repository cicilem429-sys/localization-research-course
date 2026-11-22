
import numpy as np

from localization_tools import estimate_position_multistart
from fast_geometry import fast_geometry_localization
from adaptive_ten_anchor import adaptive_ten_anchor_localization


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

rng = np.random.default_rng(2032)

for sigma in noise_levels:

    errors = {
        "Huber": [],
        "FixedGeometry": [],
        "AdaptiveGeometry": []
    }

    fixed_correct = 0
    adaptive_correct = 0
    thresholds = []

    for _ in range(trials):

        true_position = rng.uniform(
            low=[2, 2],
            high=[8, 8]
        )

        true_distances = np.linalg.norm(
            anchors - true_position,
            axis=1
        )

        distances = true_distances + rng.normal(
            0, sigma, size=len(anchors)
        )

        outlier_index = int(
            rng.integers(0, len(anchors))
        )

        distances[outlier_index] += 8.0
        distances = np.maximum(distances, 0.0)

        huber = estimate_position_multistart(
            anchors, distances,
            method="huber", delta=0.3
        )

        fixed = fast_geometry_localization(
            anchors, distances,
            threshold=1.0
        )

        adaptive = adaptive_ten_anchor_localization(
            anchors, distances,
            c=2.5,
            min_threshold=0.5
        )

        estimates = {
            "Huber": huber["position"],
            "FixedGeometry": fixed["position"],
            "AdaptiveGeometry": adaptive["position"]
        }

        for name, position in estimates.items():
            error = np.linalg.norm(
                np.asarray(position) - true_position
            )

            errors[name].append(float(error))

        if fixed["suspected_outliers"] == [
            outlier_index
        ]:
            fixed_correct += 1

        if adaptive["suspected_outliers"] == [
            outlier_index
        ]:
            adaptive_correct += 1

        thresholds.append(
            adaptive["threshold"]
        )

    print(f"\n===== 十节点 sigma={sigma} =====")

    for name, values in errors.items():

        print(f"\n{name}")

        print(
            "平均误差：",
            round(np.mean(values), 4)
        )

        print(
            "CEP90：",
            round(np.percentile(values, 90), 4)
        )

        print(
            "最大误差：",
            round(np.max(values), 4)
        )

    print(
        "\n固定阈值识别率：",
        f"{fixed_correct/trials:.1%}"
    )

    print(
        "自适应识别率：",
        f"{adaptive_correct/trials:.1%}"
    )

    print(
        "平均自适应阈值：",
        round(np.mean(thresholds), 4)
    )
