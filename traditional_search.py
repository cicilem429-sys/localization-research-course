
import json

from research_environment_v3 import (
    make_datasets,
    evaluate_strategy
)


# ========== 1. 准备实验数据 ==========

# 与 DeepSeek V3 使用相同的开发集
development_data = make_datasets(
    seed=2041,
    trials_per_case=10
)

print("开发集样本数：", len(development_data))


# ========== 2. 传统搜索的候选方案 ==========

# 预先固定三个参数配置。
# 这是一种固定候选搜索基线，
# 而不是完整网格搜索或全局优化。

candidates = [
    {
        "method": "huber",
        "parameter": 0.3
    },
    {
        "method": "fixed_geometry",
        "parameter": 2.0
    },
    {
        "method": "adaptive_geometry",
        "parameter": 2.5
    }
]


# ========== 3. 依次执行三轮实验 ==========

history = []

for index, candidate in enumerate(candidates, start=1):

    method = candidate["method"]
    parameter = candidate["parameter"]

    print(f"\n========== 传统搜索第 {index} 轮 ==========")
    print("方法：", method)
    print("参数：", parameter)

    result = evaluate_strategy(
        method=method,
        parameter=parameter,
        datasets=development_data
    )

    result["round"] = index
    history.append(result)

    print("平均定位误差：", result["mean_error"])
    print("CEP90：", result["cep90"])
    print("macro-CEP90：", result["macro_cep90"])
    print("最大定位误差：", result["max_error"])
    print(
        "异常节点识别率：",
        result["exact_outlier_detection_rate"]
    )


# ========== 4. 选出最佳候选 ==========

best = min(
    history,
    key=lambda x: x["macro_cep90"]
)

print("\n========== 传统搜索最佳候选 ==========")

print("方法：", best["method"])
print("参数：", best["parameter"])
print("macro-CEP90：", best["macro_cep90"])
print("平均误差：", best["mean_error"])


# ========== 5. 读取之前 DeepSeek 的实验结果 ==========

with open(
    "research_agent_v3_history.json",
    "r",
    encoding="utf-8"
) as f:
    agent_record = json.load(f)

agent_history = agent_record["experiments"]

agent_best = min(
    agent_history,
    key=lambda x: x["macro_cep90"]
)


# ========== 6. 比较两种研究方法 ==========

print("\n========== DeepSeek 与传统搜索比较 ==========")

print("\nDeepSeek 智能体：")
print("最佳方法：", agent_best["method"])
print("最佳参数：", agent_best["parameter"])
print("最佳 macro-CEP90：", agent_best["macro_cep90"])

print("\n传统参数搜索：")
print("最佳方法：", best["method"])
print("最佳参数：", best["parameter"])
print("最佳 macro-CEP90：", best["macro_cep90"])

difference = (
    best["macro_cep90"]
    - agent_best["macro_cep90"]
)

print("\n传统搜索分数 - DeepSeek分数：")
print(round(difference, 4))

if difference > 0:
    print("本次 DeepSeek 找到的候选方案分数更低")
elif difference < 0:
    print("本次传统搜索找到的候选方案分数更低")
else:
    print("两种方法本次最佳分数相同")


# ========== 7. 保存实验记录 ==========

record = {
    "experiment_budget": 3,
    "development_seed": 2041,
    "trials_per_case": 10,
    "traditional_experiments": history,
    "traditional_best": best,
    "agent_best": agent_best,
    "score_difference": round(difference, 4)
}

with open(
    "traditional_search_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        record,
        f,
        ensure_ascii=False,
        indent=2
    )

print(
    "\n结果已保存：traditional_search_results.json"
)
