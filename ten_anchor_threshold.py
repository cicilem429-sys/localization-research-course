
import numpy as np

from fast_geometry import fast_geometry_localization


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
thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]

trials = 50
rng = np.random.default_rng(2029)


for sigma in noise_levels:

    # 预先生成测试数据，保证不同阈值公平比较
    datasets = []

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

        datasets.append((
            true_position,
            distances,
            outlier_index
        ))

    print(f"\n===== sigma = {sigma} =====")

    for threshold in thresholds:

        errors = []
        correct = 0

        for true_position, distances, outlier_index in datasets:

            result = fast_geometry_localization(
                anchors,
                distances,
                threshold=threshold
            )

            estimated = np.asarray(
                result["position"]
            )

            errors.append(
                np.linalg.norm(
                    estimated - true_position
                )
            )

            if result["suspected_outliers"] == [
                outlier_index
            ]:
                correct += 1

        print(
            f"threshold={threshold:.1f} | "
            f"平均误差={np.mean(errors):.4f} | "
            f"CEP90={np.percentile(errors, 90):.4f} | "
            f"识别率={correct/trials:.1%}"
        )
