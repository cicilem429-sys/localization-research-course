
import numpy as np

from fast_cluster import fast_cluster_spread

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
rng = np.random.default_rng(2030)

for sigma in noise_levels:

    spreads = []

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

        result = fast_cluster_spread(
            anchors,
            distances,
            cluster_size=4
        )

        spreads.append(
            result["cluster_spread"]
        )

    print(f"\n===== sigma={sigma} =====")

    print(
        "平均聚类分散度：",
        round(np.mean(spreads), 4)
    )

    print(
        "聚类分散度标准差：",
        round(np.std(spreads), 4)
    )

    print(
        "聚类分散度中位数：",
        round(np.median(spreads), 4)
    )
