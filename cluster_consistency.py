
import numpy as np

from candidate_analysis import analyze_candidates


def candidate_cluster_spread(
    anchors,
    distances,
    cluster_size=4
):
    """
    从候选位置中寻找最紧密的 cluster_size 个点。

    这里使用最小的组内两两距离平方和作为
    聚类评分，并计算该集合的位置分散度。
    """

    from itertools import combinations

    analysis = analyze_candidates(
        anchors, distances
    )

    candidates = np.asarray(
        analysis["candidate_positions"],
        dtype=float
    )

    n = len(candidates)

    if cluster_size < 2 or cluster_size > n:
        raise ValueError("cluster_size 不合法")

    best_indices = None
    best_score = float("inf")

    for indices in combinations(range(n), cluster_size):

        points = candidates[list(indices)]

        # 组内所有点之间的平方距离之和
        diff = points[:, None, :] - points[None, :, :]

        score = np.sum(diff ** 2) / 2.0

        if score < best_score:
            best_score = score
            best_indices = indices

    selected = candidates[list(best_indices)]

    center = np.mean(selected, axis=0)

    spread = np.sqrt(
        np.mean(
            np.sum((selected - center) ** 2, axis=1)
        )
    )

    return {
        "cluster_center": center.tolist(),
        "cluster_spread": float(spread),
        "cluster_score": float(best_score),
        "selected_indices": list(best_indices),
        "candidate_subsets": [
            analysis["subsets"][i]
            for i in best_indices
        ]
    }
