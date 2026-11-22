
import numpy as np

from localization_tools import estimate_position_multistart
from fast_geometry import fast_geometry_localization


def adaptive_ten_anchor_localization(
    anchors,
    distances,
    c=2.5,
    min_threshold=0.5
):
    """
    Huber-MAD 噪声估计 +
    自适应几何一致性定位
    """

    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)

    # 第一步：使用 Huber 拟合初始位置
    huber = estimate_position_multistart(
        anchors,
        distances,
        method="huber",
        delta=0.3
    )

    position = np.asarray(
        huber["position"],
        dtype=float
    )

    # 第二步：计算有符号测距残差
    residuals = (
        np.linalg.norm(
            anchors - position,
            axis=1
        ) - distances
    )

    # 第三步：MAD 稳健噪声估计
    residual_median = np.median(residuals)

    mad = np.median(
        np.abs(residuals - residual_median)
    )

    sigma_hat = 1.4826 * mad

    # 第四步：根据估计噪声调整阈值
    threshold = max(
        min_threshold,
        c * sigma_hat
    )

    # 第五步：几何一致性定位
    geometry = fast_geometry_localization(
        anchors,
        distances,
        threshold=threshold
    )

    return {
        "position": geometry["position"],
        "sigma_hat": float(sigma_hat),
        "threshold": float(threshold),
        "suspected_outliers": geometry["suspected_outliers"],
        "inlier_count": geometry["inlier_count"]
    }
