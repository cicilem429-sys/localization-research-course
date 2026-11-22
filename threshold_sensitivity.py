
import numpy as np

from geometry_consistency import detect_outliers

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

noise_levels = [0.2, 0.5, 1.0]
thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]

trials = 100
rng = np.random.default_rng(42)

for sigma in noise_levels:
    # 先生成数据，保证各阈值使用相同的100组观测
    datasets = []

    for _ in range(trials):
        distances = true_distances + rng.normal(
            0, sigma, size=len(anchors)
        )
        distances[3] += 8.0
        distances = np.maximum(distances, 0.0)
        datasets.append(distances)

    print(f"\n===== sigma = {sigma} =====")

    for threshold in thresholds:
        errors = []
        correct = 0

        for distances in datasets:
            result = detect_outliers(
                anchors,
                distances,
                threshold=threshold
            )

            estimated = np.asarray(
                result["estimated_position"]
            )

            error = np.linalg.norm(
                estimated - true_position
            )

            errors.append(float(error))

            if result["suspected_outliers"] == [3]:
                correct += 1

        print(
            f"threshold={threshold:.1f} | "
            f"平均误差={np.mean(errors):.4f} | "
            f"最大误差={np.max(errors):.4f} | "
            f"识别率={correct / trials:.1%}"
        )
