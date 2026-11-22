
import numpy as np
from scipy.optimize import minimize

from fast_geometry import fast_geometry_localization


def weighted_huber_localization(
    anchors,
    distances,
    weights,
    delta=0.5,
    initial_position=None
):
    """
    最小化加权 Huber 目标函数：

    sum_i weights[i] * Huber_delta(residual[i])

    weights 是外部传入的可信权重。
    """

    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)
    weights = np.asarray(weights, dtype=float)

    n = len(anchors)

    if anchors.shape != (n, 2):
        raise ValueError("anchors 必须是 n×2 数组")

    if distances.shape != (n,):
        raise ValueError("distances 维度错误")

    if weights.shape != (n,):
        raise ValueError("weights 维度错误")

    if n < 3:
        raise ValueError("至少需要三个观测节点")

    if not (
        np.all(np.isfinite(anchors))
        and np.all(np.isfinite(distances))
        and np.all(np.isfinite(weights))
    ):
        raise ValueError("输入包含无效数值")

    if np.any(distances < 0):
        raise ValueError("测距不能为负")

    if np.any(weights < 0) or np.all(weights == 0):
        raise ValueError("权重必须非负且不能全为零")

    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta 必须为正数")

    def objective(position):

        residuals = (
            np.linalg.norm(
                anchors - position, axis=1
            ) - distances
        )

        abs_r = np.abs(residuals)

        huber_loss = np.where(
            abs_r <= delta,
            0.5 * residuals ** 2,
            delta * (abs_r - 0.5 * delta)
        )

        return float(
            np.sum(weights * huber_loss)
        )

    # 多起点优化，降低局部极小值的影响
    starts = [np.mean(anchors, axis=0)]

    if initial_position is not None:
        initial = np.asarray(
            initial_position, dtype=float
        )

        if initial.shape != (2,) or not np.all(
            np.isfinite(initial)
        ):
            raise ValueError("初始位置无效")

        starts.insert(0, initial)

    best = None

    for start in starts:

        result = minimize(
            objective,
            x0=start,
            method="BFGS",
            options={"maxiter": 300}
        )

        if (
            np.all(np.isfinite(result.x))
            and np.isfinite(result.fun)
            and (
                best is None
                or result.fun < best.fun
            )
        ):
            best = result

    if best is None:
        raise RuntimeError("优化未得到有限数值解")

    return {
        "position": best.x.tolist(),
        "cost": float(best.fun),
        "success": bool(best.success),
        "message": str(best.message)
    }


def geometry_weighted_localization(
    anchors,
    distances,
    threshold=1.5,
    delta=0.5,
    min_weight=0.05
):
    """
    两阶段定位：

    1. 几何一致性估计初始位置
    2. 根据初始残差构造连续权重
    3. 加权 Huber 非线性优化
    """

    if not 0 < min_weight <= 1:
        raise ValueError("min_weight 必须在 (0,1]")

    geometry = fast_geometry_localization(
        anchors,
        distances,
        threshold=threshold
    )

    initial_position = np.asarray(
        geometry["position"], dtype=float
    )

    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)

    residuals = (
        np.linalg.norm(
            anchors - initial_position, axis=1
        ) - distances
    )

    # 用连续权重替代硬删除
    weights = 1.0 / (
        1.0 + (np.abs(residuals) / threshold) ** 2
    )

    weights = np.maximum(
        weights,
        min_weight
    )

    refined = weighted_huber_localization(
        anchors,
        distances,
        weights,
        delta=delta,
        initial_position=initial_position
    )

    return {
        "position": refined["position"],
        "weights": weights.tolist(),
        "initial_position": initial_position.tolist(),
        "suspected_outliers": (
            geometry["suspected_outliers"]
        ),
        "cost": refined["cost"],
        "optimization_success": refined["success"]
    }
