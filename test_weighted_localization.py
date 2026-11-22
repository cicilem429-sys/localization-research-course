
import numpy as np

from localization_tools import estimate_position_multistart
from fast_geometry import fast_geometry_localization
from weighted_localization import (
    geometry_weighted_localization
)

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

rng = np.random.default_rng(2045)

noise_levels = [0.2, 0.5, 1.0]
trials = 20

for sigma in noise_levels:

    errors = {
        "Huber": [],
        "FixedGeometry": [],
        "WeightedGeometry": []
    }

    weight_normal = []
    weight_outlier = []

    for _ in range(trials):

        true_position = rng.uniform(
            [2, 2], [8, 8]
        )

        true_distances = np.linalg.norm(
            anchors - true_position,
            axis=1
        )

        distances = (
            true_distances
            + rng.normal(
                0, sigma, size=len(anchors)
            )
        )

        outlier_index = int(
            rng.integers(0, len(anchors))
        )

        distances[outlier_index] += 8.0
        distances = np.maximum(distances, 0.0)

        huber = estimate_position_multistart(
            anchors,
            distances,
            method="huber",
            delta=0.5
        )

        fixed = fast_geometry_localization(
            anchors,
            distances,
            threshold=1.5
        )

        weighted = geometry_weighted_localization(
            anchors,
            distances,
            threshold=1.5,
            delta=0.5
        )

        positions = {
            "Huber": huber["position"],
            "FixedGeometry": fixed["position"],
            "WeightedGeometry": weighted["position"]
        }

        for name, position in positions.items():

            error = np.linalg.norm(
                np.asarray(position)
                - true_position
            )

            errors[name].append(
                float(error)
            )

        weights = np.asarray(
            weighted["weights"]
        )

        weight_outlier.append(
            weights[outlier_index]
        )

        normal_mask = np.ones(
            len(anchors), dtype=bool
        )
        normal_mask[outlier_index] = False

        weight_normal.append(
            float(np.mean(weights[normal_mask]))
        )

    print(f"\n===== sigma={sigma} =====")

    for name, values in errors.items():

        print(f"\n{name}")

        print(
            "平均定位误差：",
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

    print("\n权重诊断：")

    print(
        "正常节点平均权重：",
        round(np.mean(weight_normal), 4)
    )

    print(
        "异常节点平均权重：",
        round(np.mean(weight_outlier), 4)
    )
