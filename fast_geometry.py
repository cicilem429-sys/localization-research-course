
import numpy as np
from itertools import combinations

from localization_tools import estimate_position_multistart


def fast_geometry_localization(
    anchors,
    distances,
    threshold=1.0
):
    """
    基于三节点解析候选解的最大一致性定位。

    与旧版本相比：
    1. 用线性方程快速生成三节点候选位置
    2. 用全部节点的残差评分
    3. 选出最大一致节点集合
    4. 使用一致节点重新进行非线性优化

    这是最大一致性思想的高效实现，
    并非全新的数学算法。
    """

    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("anchors 必须是 n×2 数组")

    n = len(anchors)

    if distances.shape != (n,):
        raise ValueError("distances 维度错误")

    if n < 4:
        raise ValueError("至少需要4个节点")

    if not np.all(np.isfinite(anchors)):
        raise ValueError("节点坐标包含非有限值")

    if not np.all(np.isfinite(distances)):
        raise ValueError("距离包含非有限值")

    if np.any(distances < 0):
        raise ValueError("测距不能为负")

    if threshold <= 0:
        raise ValueError("threshold 必须大于0")

    best_position = None
    best_inliers = None
    best_count = -1
    best_score = float("inf")

    candidate_count = 0

    for ids in combinations(range(n), 3):
        ids = list(ids)

        p = anchors[ids]
        d = distances[ids]

        # 由三个距离方程相减，
        # 得到关于二维位置的线性方程
        A = 2.0 * (p[1:] - p[0])

        b = (
            np.sum(p[1:] ** 2, axis=1)
            - np.sum(p[0] ** 2)
            + d[0] ** 2
            - d[1:] ** 2
        )

        # 跳过共线或接近共线的三节点组合
        if np.linalg.cond(A) > 1e8:
            continue

        position = np.linalg.solve(A, b)

        candidate_count += 1

        residuals = np.abs(
            np.linalg.norm(
                anchors - position, axis=1
            ) - distances
        )

        inliers = residuals <= threshold
        count = int(np.sum(inliers))

        # 一致节点越多越好
        # 数量相同则比较一致节点的残差平方和
        score = float(
            np.sum(residuals[inliers] ** 2)
        )

        if (
            count > best_count
            or (
                count == best_count
                and score < best_score
            )
        ):
            best_position = position
            best_inliers = inliers
            best_count = count
            best_score = score

    if best_position is None:
        raise RuntimeError("没有找到有效的三节点组合")

    inlier_indices = np.flatnonzero(best_inliers)

    # 使用一致节点进行非线性优化
    if len(inlier_indices) >= 3:
        result = estimate_position_multistart(
            anchors[inlier_indices],
            distances[inlier_indices],
            method="linear"
        )

        position = np.asarray(
            result["position"], dtype=float
        )
    else:
        position = best_position

    suspected_outliers = [
        i for i in range(n)
        if not best_inliers[i]
    ]

    return {
        "position": position.tolist(),
        "suspected_outliers": suspected_outliers,
        "inlier_count": int(best_count),
        "candidate_count": candidate_count
    }
