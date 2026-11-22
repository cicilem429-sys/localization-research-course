
import numpy as np

from localization_tools import estimate_position_multistart
from geometry_consistency import detect_outliers
from hybrid_localization import hybrid_localization

anchors = np.array([
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10],
    [5, 15]
], dtype=float)

true_position = np.array([4.0, 6.0])

true_distances = np.linalg.norm(
    anchors - true_position,
    axis=1
)

rng = np.random.default_rng(42)

results = {
    "OLS": [],
    "Huber_0.3": [],
    "Huber_1.0": [],
    "Geometry": [],
    "Hybrid": []
}

correct_geometry = 0
correct_hybrid = 0
trials = 100

for trial in range(trials):
    # 所有方法使用相同的随机测距数据
    distances = true_distances + rng.normal(
        0, 0.2, size=len(anchors)
    )
    distances[3] += 8.0

    ols = estimate_position_multistart(
        anchors, distances, method="linear"
    )

    huber03 = estimate_position_multistart(
        anchors, distances,
        method="huber", delta=0.3
    )

    huber10 = estimate_position_multistart(
        anchors, distances,
        method="huber", delta=1.0
    )

    geometry = detect_outliers(
        anchors, distances, threshold=1.0
    )

    hybrid = hybrid_localization(
        anchors, distances,
        threshold=1.0,
        delta=0.3
    )

    estimates = {
        "OLS": ols["position"],
        "Huber_0.3": huber03["position"],
        "Huber_1.0": huber10["position"],
        "Geometry": geometry["estimated_position"],
        "Hybrid": hybrid["position"]
    }

    for name, position in estimates.items():
        error = np.linalg.norm(
            np.asarray(position) - true_position
        )
        results[name].append(float(error))

    if geometry["suspected_outliers"] == [3]:
        correct_geometry += 1

    if hybrid["suspected_outliers"] == [3]:
        correct_hybrid += 1

print("\n===== 五种方法：100 次随机实验 =====")

for name, errors in results.items():
    print(f"\n方法：{name}")
    print(f"平均定位误差：{np.mean(errors):.6f}")
    print(f"误差标准差：{np.std(errors):.6f}")
    print(f"最大定位误差：{np.max(errors):.6f}")

print(
    "\n几何一致性异常识别率：",
    correct_geometry / trials
)

print(
    "混合方法异常识别率：",
    correct_hybrid / trials
)

