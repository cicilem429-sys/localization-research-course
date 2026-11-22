
import numpy as np

from research_environment_v3 import (
    make_datasets,
    run_strategy
)

# 使用开发数据进行探索，不使用保留集调参
datasets = make_datasets(
    seed=2050,
    trials_per_case=10
)

fixed_errors = []
adaptive_errors = []

fixed_wins = 0
adaptive_wins = 0
ties = 0

print("正在分析两种算法的互补性...", flush=True)

for item in datasets:
    anchors = item["anchors"]
    distances = item["distances"]
    true_position = item["true_position"]

    fixed = run_strategy(
        "fixed_geometry",
        1.5,
        anchors,
        distances
    )

    adaptive = run_strategy(
        "adaptive_geometry",
        2.0,
        anchors,
        distances
    )

    fixed_error = float(np.linalg.norm(
        np.asarray(fixed["position"]) - true_position
    ))

    adaptive_error = float(np.linalg.norm(
        np.asarray(adaptive["position"]) - true_position
    ))

    fixed_errors.append(fixed_error)
    adaptive_errors.append(adaptive_error)

    if fixed_error < adaptive_error - 1e-8:
        fixed_wins += 1

    elif adaptive_error < fixed_error - 1e-8:
        adaptive_wins += 1

    else:
        ties += 1


fixed_errors = np.asarray(fixed_errors)
adaptive_errors = np.asarray(adaptive_errors)

# 理想选择器：每次选实际误差更小的方法
# 注意：它偷看了真实位置，只能用于理论诊断！
oracle_errors = np.minimum(
    fixed_errors,
    adaptive_errors
)

print("\n===== 两种策略的互补性 =====")
print("样本数量：", len(datasets))
print("固定几何胜出次数：", fixed_wins)
print("自适应几何胜出次数：", adaptive_wins)
print("两种方法近似打平次数：", ties)

print("\n===== 平均定位误差 =====")
print(
    "固定几何：",
    round(np.mean(fixed_errors), 4)
)
print(
    "自适应几何：",
    round(np.mean(adaptive_errors), 4)
)
print(
    "理想选择器：",
    round(np.mean(oracle_errors), 4)
)

print("\n===== 整体 CEP90 =====")
print(
    "固定几何：",
    round(np.percentile(fixed_errors, 90), 4)
)
print(
    "自适应几何：",
    round(np.percentile(adaptive_errors, 90), 4)
)
print(
    "理想选择器：",
    round(np.percentile(oracle_errors, 90), 4)
)

best_baseline = min(
    np.mean(fixed_errors),
    np.mean(adaptive_errors)
)

improvement = (
    best_baseline - np.mean(oracle_errors)
) / best_baseline * 100

print(
    "\n理想选择器相对最佳单一方法的"
    "平均误差潜在改善：",
    f"{improvement:.2f}%"
)
