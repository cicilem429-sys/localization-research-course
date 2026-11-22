
import numpy as np

from geometry_consistency import detect_outliers
from localization_tools import estimate_position_multistart


def hybrid_localization(
    anchors,
    distances,
    threshold=1.0,
    delta=0.3
):
    """
    几何一致性筛选 + Huber 鲁棒定位
    """

    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)

    # 第一阶段：识别异常节点
    geometry = detect_outliers(
        anchors,
        distances,
        threshold=threshold
    )

    suspected = set(geometry["suspected_outliers"])

    # 保留被认为正常的节点
    inlier_indices = [
        i for i in range(len(anchors))
        if i not in suspected
    ]

    # 保证剩余节点足以进行二维定位
    if len(inlier_indices) < 3:
        raise RuntimeError("有效节点不足，无法定位")

    selected_anchors = anchors[inlier_indices]
    selected_distances = distances[inlier_indices]

    # 第二阶段：使用 Huber 损失重新定位
    refined = estimate_position_multistart(
        selected_anchors,
        selected_distances,
        method="huber",
        delta=delta
    )

    return {
        "position": refined["position"],
        "suspected_outliers": sorted(suspected),
        "inlier_count": len(inlier_indices),
        "optimization_success": refined["success"]
    }
