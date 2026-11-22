
import numpy as np

from localization_tools import estimate_position_multistart
from fast_geometry import fast_geometry_localization
from adaptive_ten_anchor import adaptive_ten_anchor_localization


# ========== 1. 实验基本设置 ==========

FIXED_ANCHORS = np.array([
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

NOISE_LEVELS = [0.2, 0.5, 1.0]

# 这些是实验场景配置，不是智能体的定位输入
SCENARIOS = {
    "standard": {
        "random_layout": False,
        "outlier_count": 1,
        "amplitude": 8.0
    },
    "random_layout": {
        "random_layout": True,
        "outlier_count": 1,
        "amplitude": 8.0
    },
    "two_outliers": {
        "random_layout": False,
        "outlier_count": 2,
        "amplitude": 8.0
    },
    "small_outlier": {
        "random_layout": False,
        "outlier_count": 1,
        "amplitude": 2.0
    }
}


# ========== 2. 生成模拟实验数据 ==========

def make_datasets(seed=2041, trials_per_case=20):
    """
    生成可复现的模拟数据。

    每条数据包含：
    - 观测节点坐标
    - 带噪声的测距
    - 仅供评价使用的真实位置及异常节点编号
    """

    if trials_per_case < 1:
        raise ValueError("实验次数必须大于零")

    rng = np.random.default_rng(seed)
    datasets = []

    for scenario_name, config in SCENARIOS.items():

        for sigma in NOISE_LEVELS:

            for _ in range(trials_per_case):

                if config["random_layout"]:
                    anchors = rng.uniform(
                        -5.0, 15.0, size=(10, 2)
                    )
                else:
                    anchors = FIXED_ANCHORS.copy()

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

                outlier_indices = rng.choice(
                    len(anchors),
                    size=config["outlier_count"],
                    replace=False
                )

                distances[outlier_indices] += (
                    config["amplitude"]
                )

                distances = np.maximum(
                    distances, 0.0
                )

                datasets.append({
                    "scenario": scenario_name,
                    "sigma": sigma,
                    "anchors": anchors,
                    "distances": distances,
                    "true_position": true_position,
                    "outlier_indices": sorted(
                        outlier_indices.tolist()
                    )
                })

    return datasets


# ========== 3. 执行单次定位 ==========

def run_strategy(method, parameter, anchors, distances):
    """
    只接受节点坐标和观测距离。
    不接受真实目标位置或真实异常节点编号。
    """

    parameter = float(parameter)

    if not np.isfinite(parameter):
        raise ValueError("参数必须为有限数值")

    if method == "huber":

        if not 0.1 <= parameter <= 4.0:
            raise ValueError(
                "Huber delta 必须在 0.1 到 4.0"
            )

        result = estimate_position_multistart(
            anchors,
            distances,
            method="huber",
            delta=parameter
        )

        return {
            "position": result["position"],
            "suspected_outliers": None
        }

    elif method == "fixed_geometry":

        if not 0.3 <= parameter <= 4.0:
            raise ValueError(
                "固定几何阈值必须在 0.3 到 4.0"
            )

        result = fast_geometry_localization(
            anchors,
            distances,
            threshold=parameter
        )

        return {
            "position": result["position"],
            "suspected_outliers": (
                result["suspected_outliers"]
            )
        }

    elif method == "adaptive_geometry":

        if not 0.5 <= parameter <= 5.0:
            raise ValueError(
                "自适应系数 c 必须在 0.5 到 5.0"
            )

        result = adaptive_ten_anchor_localization(
            anchors,
            distances,
            c=parameter,
            min_threshold=0.5
        )

        return {
            "position": result["position"],
            "suspected_outliers": (
                result["suspected_outliers"]
            )
        }

    else:
        raise ValueError("不支持的定位方法：" + str(method))


# ========== 4. 评价一种策略 ==========

def evaluate_strategy(method, parameter, datasets):
    """
    使用相同数据评价指定策略。

    真实位置仅在此处用于计算实验指标，
    不会传入 run_strategy。
    """

    # 先检查参数，避免无效配置运行整个数据集
    if method not in (
        "huber",
        "fixed_geometry",
        "adaptive_geometry"
    ):
        raise ValueError("未知方法")

    if len(datasets) == 0:
        raise ValueError("数据集不能为空")

    errors = []
    grouped_errors = {}

    exact_correct = 0

    for item in datasets:

        result = run_strategy(
            method,
            parameter,
            item["anchors"],
            item["distances"]
        )

        estimated = np.asarray(
            result["position"], dtype=float
        )

        if estimated.shape != (2,):
            raise RuntimeError("定位结果不是二维坐标")

        error = float(np.linalg.norm(
            estimated - item["true_position"]
        ))

        if not np.isfinite(error):
            raise RuntimeError("出现非有限定位误差")

        errors.append(error)

        key = (
            item["scenario"],
            float(item["sigma"])
        )

        grouped_errors.setdefault(key, [])
        grouped_errors[key].append(error)

        suspected = result["suspected_outliers"]

        if suspected is not None:
            if sorted(suspected) == item["outlier_indices"]:
                exact_correct += 1

    per_case = {}

    case_cep90_values = []

    for (scenario, sigma), values in grouped_errors.items():

        cep90 = float(np.percentile(values, 90))
        case_cep90_values.append(cep90)

        case_name = f"{scenario}|sigma={sigma}"

        per_case[case_name] = {
            "sample_count": len(values),
            "mean_error": round(
                float(np.mean(values)), 4
            ),
            "cep90": round(cep90, 4)
        }

    # 对每个场景-噪声组合给予相同权重
    macro_cep90 = float(np.mean(case_cep90_values))

    return {
        "method": method,
        "parameter": float(parameter),
        "sample_count": len(errors),
        "mean_error": round(
            float(np.mean(errors)), 4
        ),
        "cep90": round(
            float(np.percentile(errors, 90)), 4
        ),
        "macro_cep90": round(macro_cep90, 4),
        "max_error": round(float(np.max(errors)), 4),
        "exact_outlier_detection_rate": (
            None if method == "huber"
            else round(exact_correct / len(datasets), 4)
        ),
        "per_case": per_case
    }


# ========== 5. 本地测试入口 ==========

if __name__ == "__main__":

    print("正在生成 V3 实验数据...", flush=True)

    datasets = make_datasets(
        seed=2041,
        trials_per_case=5
    )

    print("数据总数：", len(datasets), flush=True)

    print("\n正在测试 Huber 方法...", flush=True)

    result = evaluate_strategy(
        method="huber",
        parameter=1.0,
        datasets=datasets
    )

    print("\n===== V3 环境测试成功 =====")
    print("方法：", result["method"])
    print("样本数：", result["sample_count"])
    print("平均定位误差：", result["mean_error"])
    print("CEP90：", result["cep90"])
    print("macro-CEP90：", result["macro_cep90"])
    print("最大定位误差：", result["max_error"])
    print("场景数量：", len(result["per_case"]))
