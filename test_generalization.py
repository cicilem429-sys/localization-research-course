
import json
import numpy as np

from localization_tools import estimate_position_multistart
from fast_geometry import fast_geometry_localization
from adaptive_ten_anchor import adaptive_ten_anchor_localization


# 原来的十节点布局
fixed_anchors = np.array([
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

# 之前智能体测试过的三种策略
strategies = {
    "Huber": 1.0,
    "FixedGeometry": 3.0,
    "AdaptiveGeometry": 2.0
}

noise_levels = [0.2, 0.5, 1.0]
trials = 30

rng = np.random.default_rng(2040)

scenarios = {
    "A_random_layout": {
        "random_layout": True,
        "outlier_count": 1,
        "amplitude": 8.0
    },
    "B_two_outliers": {
        "random_layout": False,
        "outlier_count": 2,
        "amplitude": 8.0
    },
    "C_small_outlier": {
        "random_layout": False,
        "outlier_count": 1,
        "amplitude": 2.0
    }
}


def generate_datasets(config, sigma):
    """生成一批固定数据，供所有方法公平比较。"""

    datasets = []

    for _ in range(trials):

        if config["random_layout"]:
            anchors = rng.uniform(
                low=-5.0,
                high=15.0,
                size=(10, 2)
            )
        else:
            anchors = fixed_anchors.copy()

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

        outlier_indices = rng.choice(
            len(anchors),
            size=config["outlier_count"],
            replace=False
        )

        distances[outlier_indices] += config["amplitude"]
        distances = np.maximum(distances, 0.0)

        datasets.append({
            "anchors": anchors,
            "distances": distances,
            "true_position": true_position,
            "outlier_indices": sorted(
                outlier_indices.tolist()
            )
        })

    return datasets


def run_method(name, parameter, item):
    """运行某一种定位策略。"""

    anchors = item["anchors"]
    distances = item["distances"]

    if name == "Huber":

        result = estimate_position_multistart(
            anchors,
            distances,
            method="huber",
            delta=parameter
        )

        return result["position"], None

    if name == "FixedGeometry":

        result = fast_geometry_localization(
            anchors,
            distances,
            threshold=parameter
        )

        return (
            result["position"],
            result["suspected_outliers"]
        )

    result = adaptive_ten_anchor_localization(
        anchors,
        distances,
        c=parameter,
        min_threshold=0.5
    )

    return (
        result["position"],
        result["suspected_outliers"]
    )


report = []

for scenario_name, config in scenarios.items():

    print(f"\n========== 场景：{scenario_name} ==========")

    for sigma in noise_levels:

        # 每个方法都使用完全相同的数据
        datasets = generate_datasets(config, sigma)

        print(f"\n--- 正常噪声 sigma={sigma} ---")

        for method, parameter in strategies.items():

            errors = []
            correct = 0

            for item in datasets:

                position, suspected = run_method(
                    method,
                    parameter,
                    item
                )

                error = float(
                    np.linalg.norm(
                        np.asarray(position)
                        - item["true_position"]
                    )
                )

                if not np.isfinite(error):
                    raise RuntimeError(
                        "定位结果出现非有限误差"
                    )

                errors.append(error)

                if suspected == item["outlier_indices"]:
                    correct += 1

            mean_error = float(np.mean(errors))
            cep90 = float(np.percentile(errors, 90))

            detection_rate = (
                None if method == "Huber"
                else correct / len(datasets)
            )

            result = {
                "scenario": scenario_name,
                "sigma": sigma,
                "method": method,
                "parameter": parameter,
                "sample_count": len(datasets),
                "mean_error": round(mean_error, 4),
                "cep90": round(cep90, 4),
                "detection_rate": (
                    None if detection_rate is None
                    else round(detection_rate, 4)
                )
            }

            report.append(result)

            print(
                f"{method}: "
                f"平均误差={mean_error:.4f}, "
                f"CEP90={cep90:.4f}, "
                f"异常识别率={detection_rate}"
            )

with open(
    "generalization_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        report,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n泛化实验结果已保存：generalization_results.json")
