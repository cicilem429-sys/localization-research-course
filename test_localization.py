
import numpy as np
from localization_tools import estimate_position

# 四个已知观测节点
anchors = [
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10]
]

# 模拟实验中已知的真实目标位置
true_position = np.array([4.0, 6.0])

# 计算理想测距
distances = np.linalg.norm(
    np.asarray(anchors) - true_position,
    axis=1
)

# 人为制造一个异常观测
distances[3] += 8.0

# 两种方法分别定位
ols = estimate_position(
    anchors, distances, method="linear"
)

huber = estimate_position(
    anchors, distances, method="huber", delta=1.0
)

print("真实位置：", true_position.tolist())
print("测量距离：", distances.tolist())

for name, result in [
    ("普通最小二乘", ols),
    ("Huber 鲁棒定位", huber)
]:
    position = np.array(result["position"])
    error = np.linalg.norm(position - true_position)

    print(f"\n===== {name} =====")
    print("估计位置：", position.tolist())
    print("定位误差：", error)
    print("优化成功：", result["success"])
