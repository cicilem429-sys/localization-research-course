
import numpy as np
from itertools import combinations

from localization_tools import estimate_position_multistart


def analyze_candidates(anchors, distances):
    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)

    candidates = []
    subsets = []

    for indices in combinations(range(len(anchors)), 3):
        selected_anchors = anchors[list(indices)]

        # 排除共线三点
        vectors = selected_anchors[1:] - selected_anchors[0]

        if abs(np.linalg.det(vectors)) < 1e-10:
            continue

        selected_distances = distances[list(indices)]

        result = estimate_position_multistart(
            selected_anchors,
            selected_distances,
            method="linear"
        )

        if not result["success"]:
            continue

        position = np.asarray(
            result["position"],
            dtype=float
        )

        if not np.all(np.isfinite(position)):
            continue

        candidates.append(position)
        subsets.append(indices)

    if not candidates:
        raise RuntimeError("未找到有效候选位置")

    candidates = np.asarray(candidates)

    center = np.median(candidates, axis=0)

    distances_to_center = np.linalg.norm(
        candidates - center,
        axis=1
    )

    spread = np.median(distances_to_center)

    return {
        "candidate_positions": candidates.tolist(),
        "subsets": [list(s) for s in subsets],
        "candidate_center": center.tolist(),
        "candidate_spread": float(spread),
        "candidate_count": len(candidates)
    }
