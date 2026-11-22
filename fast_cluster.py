
import numpy as np
from itertools import combinations


def generate_candidates(anchors, distances):
    """利用三节点测距方程生成候选位置。"""

    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)

    candidates = []

    for ids in combinations(range(len(anchors)), 3):
        p = anchors[list(ids)]
        d = distances[list(ids)]

        A = 2.0 * (p[1:] - p[0])

        b = (
            np.sum(p[1:] ** 2, axis=1)
            - np.sum(p[0] ** 2)
            + d[0] ** 2
            - d[1:] ** 2
        )

        if np.linalg.cond(A) > 1e8:
            continue

        position = np.linalg.solve(A, b)
        candidates.append(position)

    if not candidates:
        raise RuntimeError("没有有效候选位置")

    return np.asarray(candidates)


def fast_cluster_spread(
    anchors,
    distances,
    cluster_size=4
):
    """计算近邻候选集合的位置分散度。"""

    candidates = generate_candidates(
        anchors, distances
    )

    n = len(candidates)

    if cluster_size < 2 or cluster_size > n:
        raise ValueError("cluster_size 不合法")

    best_spread = float("inf")
    best_center = None

    for i in range(n):

        distances_to_i = np.linalg.norm(
            candidates - candidates[i],
            axis=1
        )

        nearest = np.argsort(
            distances_to_i
        )[:cluster_size]

        selected = candidates[nearest]

        center = np.mean(selected, axis=0)

        spread = np.sqrt(
            np.mean(
                np.sum(
                    (selected - center) ** 2,
                    axis=1
                )
            )
        )

        if spread < best_spread:
            best_spread = spread
            best_center = center

    return {
        "cluster_spread": float(best_spread),
        "cluster_center": best_center.tolist(),
        "candidate_count": n
    }
