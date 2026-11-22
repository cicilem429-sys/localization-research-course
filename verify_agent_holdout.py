
import json
import numpy as np

from localization_tools import estimate_position_multistart
from fast_geometry import fast_geometry_localization
from adaptive_ten_anchor import adaptive_ten_anchor_localization

anchors = np.array([
    [0, 0], [10, 0], [0, 10], [10, 10],
    [5, 15], [15, 5], [-5, 5], [5, -5],
    [15, 15], [-5, 15]
], dtype=float)

noise_levels = [0.2, 0.5, 1.0]
amplitudes = [4.0, 8.0]

# 与智能体 V2 的独立测试集使用相同随机种子
rng = np.random.default_rng(3033)

datasets = []

for sigma in noise_levels:
    for amplitude in amplitudes:
        for _ in range(20):
            true_position = rng.uniform(
                low=[2, 2], high=[8, 8]
            )

            true_distances = np.linalg.norm(
                anchors - true_position, axis=1
            )

            distances = true_distances + rng.normal(
                0, sigma, size=len(anchors)
            )

            outlier_index = int(
                rng.integers(0, len(anchors))
            )

            distances[outlier_index] += amplitude
            distances = np.maximum(distances, 0.0)

            datasets.append({
                "true_position": true_position,
                "distances": distances,
                "outlier_index": outlier_index,
                "sigma": sigma
            })

strategies = [
    ("Huber", 1.0),
    ("FixedGeometry", 3.0),
    ("AdaptiveGeometry", 2.0)
]

report = []

for name, parameter in strategies:
    errors = []
    per_noise = {s: [] for s in noise_levels}
    correct = 0

    for item in datasets:
        distances = item["distances"]

        if name == "Huber":
            result = estimate_position_multistart(
                anchors, distances,
                method="huber", delta=parameter
            )
            position = result["position"]
            suspected = None

        elif name == "FixedGeometry":
            result = fast_geometry_localization(
                anchors, distances,
                threshold=parameter
            )
            position = result["position"]
            suspected = result["suspected_outliers"]

        else:
            result = adaptive_ten_anchor_localization(
                anchors, distances,
                c=parameter,
                min_threshold=0.5
            )
            position = result["position"]
            suspected = result["suspected_outliers"]

        error = float(np.linalg.norm(
            np.asarray(position) - item["true_position"]
        ))

        errors.append(error)
        per_noise[item["sigma"]].append(error)

        if suspected == [item["outlier_index"]]:
            correct += 1

    macro_cep90 = np.mean([
        np.percentile(per_noise[s], 90)
        for s in noise_levels
    ])

    item_result = {
        "method": name,
        "parameter": parameter,
        "mean_error": round(float(np.mean(errors)), 4),
        "cep90": round(float(np.percentile(errors, 90)), 4),
        "macro_cep90": round(float(macro_cep90), 4),
        "detection_rate": (
            None if name == "Huber"
            else round(correct / len(datasets), 4)
        )
    }

    report.append(item_result)

    print(f"\n===== {name} =====")
    print("平均误差：", item_result["mean_error"])
    print("CEP90：", item_result["cep90"])
    print("macro-CEP90：", item_result["macro_cep90"])
    print("异常识别率：", item_result["detection_rate"])

with open(
    "agent_holdout_comparison.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n独立测试对比已保存")
