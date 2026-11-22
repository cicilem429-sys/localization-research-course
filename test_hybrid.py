
import numpy as np

from hybrid_localization import hybrid_localization

anchors = np.array([
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10],
    [5, 15]
], dtype=float)

true_position = np.array([4.0, 6.0])

distances = np.linalg.norm(
    anchors - true_position,
    axis=1
)

rng = np.random.default_rng(42)

# 添加正常测距噪声
distances += rng.normal(
    0, 0.2, size=len(anchors)
)

# 第四个节点添加异常误差
distances[3] += 8.0

result = hybrid_localization(
    anchors,
    distances,
    threshold=1.0,
    delta=0.3
)

estimated = np.array(result["position"])
error = np.linalg.norm(estimated - true_position)

print("真实位置：", true_position.tolist())
print("混合方法估计位置：", result["position"])
print("实际定位误差：", error)
print("可疑异常节点：", result["suspected_outliers"])
print("有效节点数量：", result["inlier_count"])
print("优化成功：", result["optimization_success"])
