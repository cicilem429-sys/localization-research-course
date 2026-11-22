
import numpy as np
from scipy.optimize import least_squares


def estimate_position(anchors, distances, method="linear", delta=1.0):
    """
    二维测距定位。

    anchors: 已知节点坐标，例如 [[0,0], [10,0], [0,10]]
    distances: 每个节点测量得到的距离
    method: "linear" 表示普通平方损失
            "huber" 表示 Huber 鲁棒损失
    delta: Huber 损失阈值
    """

    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("anchors 必须是 n×2 坐标数组")

    if len(anchors) < 3:
        raise ValueError("至少需要三个观测节点")

    if distances.shape != (len(anchors),):
        raise ValueError("距离数量必须与节点数量一致")

    if not np.isfinite(anchors).all():
        raise ValueError("节点坐标必须是有限数值")

    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("测量距离必须是非负有限数值")

    if np.linalg.matrix_rank(anchors - anchors.mean(axis=0)) < 2:
        raise ValueError("观测节点不能全部共线")

    if method not in ("linear", "huber"):
        raise ValueError("method 必须是 linear 或 huber")

    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta 必须为正的有限数")

    # 定义测距残差
    def residuals(position):
        predicted = np.linalg.norm(
            anchors - position, axis=1
        )
        return predicted - distances

    # 使用观测节点的平均位置作为初始点
    initial_position = anchors.mean(axis=0)

    result = least_squares(
        residuals,
        x0=initial_position,
        loss=method,
        f_scale=delta
    )

    return {
        "position": result.x.tolist(),
        "cost": float(result.cost),
        "success": bool(result.success),
        "message": result.message
    }


def estimate_position_multistart(
    anchors, distances, method="huber", delta=1.0
):
    """
    从多个初始位置求解二维定位问题，
    返回目标损失最小的结果。
    """
    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("anchors 必须为 n×2 数组")
    if len(anchors) < 3 or distances.shape != (len(anchors),):
        raise ValueError("节点或测距数量不合法")
    if not np.isfinite(anchors).all():
        raise ValueError("节点坐标必须为有限数值")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("测距必须为非负有限数值")
    if np.linalg.matrix_rank(
        anchors - anchors.mean(axis=0)
    ) < 2:
        raise ValueError("观测节点不能全部共线")
    if method not in ("linear", "huber"):
        raise ValueError("不支持的损失函数")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta 必须为正数")

    def residuals(position):
        return np.linalg.norm(
            anchors - position, axis=1
        ) - distances

    initial_points = [
        anchors.mean(axis=0),
        *anchors
    ]

    best = None

    for initial in initial_points:
        result = least_squares(
            residuals,
            x0=initial,
            loss=method,
            f_scale=delta
        )

        if result.success and (
            best is None or result.cost < best.cost
        ):
            best = result

    if best is None:
        raise RuntimeError("所有初始点的优化均未成功")

    return {
        "position": best.x.tolist(),
        "cost": float(best.cost),
        "success": bool(best.success)
    }
