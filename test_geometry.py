
import numpy as np

from geometry_consistency import detect_outliers

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

# 第四个节点的测距人为增加 8
distances[3] += 8.0

result = detect_outliers(
    anchors,
    distances,
    threshold=1.0
)

estimated = np.array(result["estimated_position"])
error = np.linalg.norm(estimated - true_position)

print("真实位置：", true_position.tolist())
print("估计位置：", result["estimated_position"])
print("实际定位误差：", error)
print("一致节点数量：", result["inlier_count"])
print("可疑异常节点索引：", result["suspected_outliers"])
print("各节点残差：", result["residuals"])
print("候选位置数量：", result["candidate_count"])

print(
    "可疑异常节点编号（从1开始）：",
    [i + 1 for i in result["suspected_outliers"]]
)
