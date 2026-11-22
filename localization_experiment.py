
import numpy as np
from localization_tools import estimate_position_multistart

# 五个观测节点，避免留一后节点不足
ANCHORS = np.array([
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10],
    [5, 15]
], dtype=float)

# 模拟真实位置，仅用于生成数据和最终评价
TRUE_POSITION = np.array([4.0, 6.0])

DISTANCES = np.linalg.norm(
    ANCHORS - TRUE_POSITION, axis=1
)

# 第四个节点受到异常测距干扰
DISTANCES[3] += 8.0


def run_localization_experiment(delta):
    """
    使用留一观测交叉验证评价 Huber 阈值。
    返回给智能体的结果不包含真实位置。
    """
    delta = float(delta)

    if not np.isfinite(delta) or not 0.1 <= delta <= 10:
        raise ValueError("delta 必须在 0.1 到 10 之间")

    validation_errors = []

    for i in range(len(ANCHORS)):
        mask = np.arange(len(ANCHORS)) != i

        result = estimate_position_multistart(
            ANCHORS[mask],
            DISTANCES[mask],
            method="huber",
            delta=delta
        )

        estimated = np.asarray(result["position"])

        predicted_distance = np.linalg.norm(
            estimated - ANCHORS[i]
        )

        error = abs(predicted_distance - DISTANCES[i])
        validation_errors.append(float(error))

    # 中位数比均值更不容易被单个大误差主导
    validation_score = float(np.median(validation_errors))

    # 全部观测用于最终位置估计
    full_result = estimate_position_multistart(
        ANCHORS,
        DISTANCES,
        method="huber",
        delta=delta
    )

    return {
        "delta": delta,
        "validation_score": validation_score,
        "validation_errors": validation_errors,
        "estimated_position": full_result["position"],
        "optimization_success": full_result["success"]
    }


if __name__ == "__main__":
    for delta in [0.3, 1.0, 2.0]:
        print(run_localization_experiment(delta))
