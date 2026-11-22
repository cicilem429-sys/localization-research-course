
import numpy as np

from geometry_consistency import detect_outliers


def estimate_noise(anchors, distances, threshold=1.0):
    """
    使用几何一致性估计位置，再基于残差的MAD估计噪声。

    注意：
    这是一个待验证的估计器。
    返回的 sigma_hat 不保证等于真实噪声标准差。
    """

    anchors = np.asarray(anchors, dtype=float)
    distances = np.asarray(distances, dtype=float)

    result = detect_outliers(
        anchors,
        distances,
        threshold=threshold
    )

    position = np.asarray(
        result["estimated_position"],
        dtype=float
    )

    # 计算全部节点的有符号测距残差
    predicted_distances = np.linalg.norm(
        anchors - position,
        axis=1
    )

    residuals = predicted_distances - distances

    # 只使用被判为正常的节点估计噪声
    suspected = set(result["suspected_outliers"])

    inlier_indices = [
        i for i in range(len(anchors))
        if i not in suspected
    ]

    inlier_residuals = residuals[inlier_indices]

    # 计算中位数绝对偏差
    center = np.median(inlier_residuals)

    mad = np.median(
        np.abs(inlier_residuals - center)
    )

    sigma_hat = 1.4826 * mad

    return {
        "sigma_hat": float(sigma_hat),
        "mad": float(mad),
        "estimated_position": position.tolist(),
        "suspected_outliers": sorted(suspected),
        "inlier_count": len(inlier_indices)
    }
