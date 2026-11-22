
import numpy as np

from cluster_consistency import candidate_cluster_spread
from geometry_consistency import detect_outliers


def adaptive_localization(
    anchors,
    distances,
    a=1.0515,
    c=2.5,
    min_threshold=0.5
):
    """
    利用候选集合分散度估计噪声，
    自适应设置几何一致性阈值。
    """

    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)

    # 第一步：候选位置聚类
    cluster = candidate_cluster_spread(
        anchors,
        distances,
        cluster_size=4
    )

    spread = cluster["cluster_spread"]

    # 第二步：估计噪声
    sigma_hat = a * spread

    # 第三步：计算自适应阈值
    threshold = max(
        min_threshold,
        c * sigma_hat
    )

    # 第四步：几何一致性定位
    result = detect_outliers(
        anchors,
        distances,
        threshold=threshold
    )

    return {
        "position": result["estimated_position"],
        "sigma_hat": float(sigma_hat),
        "threshold": float(threshold),
        "suspected_outliers": result["suspected_outliers"],
        "candidate_spread": float(spread)
    }
