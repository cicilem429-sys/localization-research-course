
import numpy as np

from candidate_analysis import analyze_candidates

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

rng = np.random.default_rng(42)

for sigma in noise_levels:
    spreads = []

    for _ in range(trials):
        distances = true_distances + rng.normal(
            0, sigma, size=len(anchors)
        )

        distances[3] += 8.0

        analysis = analyze_candidates(
            anchors, distances
        )

        spreads.append(
            analysis["candidate_spread"]
        )

    print(f"\n===== 真实 sigma = {sigma} =====")

    print(
        "平均候选位置分散度：",
        round(np.mean(spreads), 4)
    )

    print(
        "分散度标准差：",
        round(np.std(spreads), 4)
    )

    print(
        "分散度中位数：",
        round(np.median(spreads), 4)
    )

    print(
        "最大分散度：",
        round(np.max(spreads), 4)
    )
