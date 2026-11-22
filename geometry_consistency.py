
import itertools
import numpy as np

from localization_tools import estimate_position_multistart


def detect_outliers(anchors, distances, threshold=1.0):
    """
    基于三节点组合的几何一致性检测。

    threshold: 判断测距是否一致的残差阈值
    """
    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("anchors 必须是 n×2 数组")

    n = len(anchors)

    if n < 4:
        raise ValueError("至少需要四个观测节点")

    if distances.shape != (n,):
        raise ValueError("测距数量与节点数量不一致")

    if not np.isfinite(anchors).all():
        raise ValueError("节点坐标必须是有限数值")

    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("测距必须是非负有限数值")

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold 必须是正数")

    candidates = []

    # 遍历所有三个节点的组合
    for indices in itertools.combinations(range(n), 3):
        subset_anchors = anchors[list(indices)]
        subset_distances = distances[list(indices)]

        # 跳过共线节点组合
        centered = subset_anchors - subset_anchors.mean(axis=0)
        if np.linalg.matrix_rank(centered) < 2:
            continue

        result = estimate_position_multistart(
            subset_anchors,
            subset_distances,
            method="linear"
        )

        position = np.asarray(result["position"])

        # 用全部节点评价这个候选位置
        predicted = np.linalg.norm(
            anchors - position,
            axis=1
        )

        residuals = np.abs(predicted - distances)

        # 一致节点数量越多越好
        inliers = residuals <= threshold
        inlier_count = int(np.sum(inliers))

        # 一致节点内部的残差越小越好
        inlier_error = float(np.sum(residuals[inliers] ** 2))

        candidates.append({
            "position": position.tolist(),
            "inlier_count": inlier_count,
            "inlier_error": inlier_error,
            "inliers": inliers.tolist(),
            "residuals": residuals.tolist()
        })

    if not candidates:
        raise RuntimeError("没有有效的三节点组合")

    # 优先最大一致节点数，再比较一致节点残差
    best = min(
        candidates,
        key=lambda c: (
            -c["inlier_count"],
            c["inlier_error"]
        )
    )

    suspected_outliers = [
        i for i, is_inlier in enumerate(best["inliers"])
        if not is_inlier
    ]

    return {
        "estimated_position": best["position"],
        "inlier_count": best["inlier_count"],
        "suspected_outliers": suspected_outliers,
        "residuals": best["residuals"],
        "candidate_count": len(candidates)
    }
