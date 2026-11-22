
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
true_distances = np.linalg.norm(
    anchors - true_position, axis=1
)

rng = np.random.default_rng(42)

errors = []
correct_detections = 0
successes = 0

for trial in range(100):
    # 所有节点都存在正常测距噪声
    distances = true_distances + rng.normal(
        0, 0.2, size=len(anchors)
    )

    # 第四个节点额外受到异常干扰
    distances[3] += 8.0

    result = detect_outliers(
        anchors,
        distances,
        threshold=1.0
    )

    estimated = np.array(result["estimated_position"])
    error = np.linalg.norm(estimated - true_position)

    errors.append(error)
    successes += 1

    if result["suspected_outliers"] == [3]:
        correct_detections += 1

print("实验次数：", successes)
print("平均定位误差：", np.mean(errors))
print("定位误差标准差：", np.std(errors))
print("最大定位误差：", np.max(errors))
print(
    "准确识别异常节点的比例：",
    correct_detections / successes
)
