
import json

from research_environment_v3 import (
    make_datasets,
    evaluate_strategy
)

# 与 V3 保留集使用完全相同的随机种子
holdout_data = make_datasets(
    seed=9041,
    trials_per_case=10
)

# DeepSeek 在 V3 三轮中选择过的方案
strategies = [
    ("huber", 1.0),
    ("fixed_geometry", 1.5),
    ("adaptive_geometry", 2.0)
]

results = []

print("===== V3 保留集公平比较 =====")
print("样本数量：", len(holdout_data))

for method, parameter in strategies:

    result = evaluate_strategy(
        method,
        parameter,
        holdout_data
    )

    results.append(result)

    print(f"\n方法：{method}")
    print("参数：", parameter)
    print("平均定位误差：", result["mean_error"])
    print("CEP90：", result["cep90"])
    print("macro-CEP90：", result["macro_cep90"])
    print("最大定位误差：", result["max_error"])
    print(
        "异常识别率：",
        result["exact_outlier_detection_rate"]
    )

# 对三种候选方案排序
ranking = sorted(
    results,
    key=lambda x: x["macro_cep90"]
)

print("\n===== 保留集排名 =====")

for i, result in enumerate(ranking, start=1):
    print(
        f"{i}. {result['method']} "
        f"参数={result['parameter']} "
        f"macro-CEP90={result['macro_cep90']}"
    )

# 保存结果
with open(
    "v3_holdout_comparison.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        results,
        file,
        ensure_ascii=False,
        indent=2
    )

print("\n结果已保存至 v3_holdout_comparison.json")
