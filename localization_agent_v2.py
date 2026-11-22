
import os
import json
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

from localization_tools import estimate_position_multistart
from fast_geometry import fast_geometry_localization
from adaptive_ten_anchor import adaptive_ten_anchor_localization


# ========== 1. 连接 DeepSeek ==========

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "请检查 .env 中是否配置 DEEPSEEK_API_KEY"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    timeout=90.0
)


# ========== 2. 十节点数学模型 ==========

anchors = np.array([
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10],
    [5, 15],
    [15, 5],
    [-5, 5],
    [5, -5],
    [15, 15],
    [-5, 15]
], dtype=float)

noise_levels = [0.2, 0.5, 1.0]
outlier_amplitudes = [4.0, 8.0]


def make_datasets(seed):
    """
    生成固定、可复现的模拟实验数据。

    注意：真实位置只用于计算实验评价指标，
    不作为定位算法的输入。
    """

    rng = np.random.default_rng(seed)
    datasets = []

    for sigma in noise_levels:
        for amplitude in outlier_amplitudes:
            for _ in range(20):

                true_position = rng.uniform(
                    low=[2, 2],
                    high=[8, 8]
                )

                true_distances = np.linalg.norm(
                    anchors - true_position,
                    axis=1
                )

                distances = (
                    true_distances
                    + rng.normal(
                        0, sigma, size=len(anchors)
                    )
                )

                outlier_index = int(
                    rng.integers(0, len(anchors))
                )

                distances[outlier_index] += amplitude
                distances = np.maximum(distances, 0.0)

                datasets.append({
                    "true_position": true_position,
                    "distances": distances,
                    "outlier_index": outlier_index,
                    "sigma": sigma,
                    "amplitude": amplitude
                })

    return datasets


# 用于智能体比较和调整方法
development_data = make_datasets(2033)

# 独立随机测试数据，不提供给 DeepSeek 迭代使用
holdout_data = make_datasets(3033)


# ========== 3. 数学实验工具 ==========

def evaluate_strategy(method, parameter, datasets):
    """
    比较一种定位策略在同一批模拟数据上的表现。
    """

    method = str(method)
    parameter = float(parameter)

    if not np.isfinite(parameter):
        raise ValueError("参数必须是有限数值")

    if method not in [
        "huber",
        "fixed_geometry",
        "adaptive_geometry"
    ]:
        raise ValueError("不支持的定位方法")

    if method == "huber":
        if not 0.1 <= parameter <= 3.0:
            raise ValueError("Huber delta 必须在 0.1 到 3.0")

    elif method == "fixed_geometry":
        if not 0.3 <= parameter <= 4.0:
            raise ValueError("几何阈值必须在 0.3 到 4.0")

    elif method == "adaptive_geometry":
        if not 0.5 <= parameter <= 5.0:
            raise ValueError("自适应系数 c 必须在 0.5 到 5.0")

    grouped_errors = {
        sigma: [] for sigma in noise_levels
    }

    all_errors = []
    correct = 0

    for item in datasets:

        distances = item["distances"]

        if method == "huber":

            result = estimate_position_multistart(
                anchors,
                distances,
                method="huber",
                delta=parameter
            )

            position = result["position"]
            suspected = None

        elif method == "fixed_geometry":

            result = fast_geometry_localization(
                anchors,
                distances,
                threshold=parameter
            )

            position = result["position"]
            suspected = result["suspected_outliers"]

        else:

            result = adaptive_ten_anchor_localization(
                anchors,
                distances,
                c=parameter,
                min_threshold=0.5
            )

            position = result["position"]
            suspected = result["suspected_outliers"]

        error = np.linalg.norm(
            np.asarray(position, dtype=float)
            - item["true_position"]
        )

        if not np.isfinite(error):
            raise RuntimeError("出现非有限定位误差")

        all_errors.append(float(error))
        grouped_errors[item["sigma"]].append(
            float(error)
        )

        if suspected == [item["outlier_index"]]:
            correct += 1

    per_noise = {}

    for sigma, values in grouped_errors.items():
        per_noise[str(sigma)] = {
            "mean_error": round(
                float(np.mean(values)), 4
            ),
            "cep90": round(
                float(np.percentile(values, 90)), 4
            )
        }

    # 按不同噪声场景的 CEP90 计算总体评分
    macro_cep90 = float(
        np.mean([
            per_noise[str(sigma)]["cep90"]
            for sigma in noise_levels
        ])
    )

    return {
        "method": method,
        "parameter": parameter,
        "sample_count": len(datasets),
        "mean_error": round(
            float(np.mean(all_errors)), 4
        ),
        "cep90": round(
            float(np.percentile(all_errors, 90)), 4
        ),
        "macro_cep90": round(macro_cep90, 4),
        "per_noise": per_noise,
        "exact_outlier_detection_rate": (
            None if method == "huber"
            else round(correct / len(datasets), 4)
        )
    }


# ========== 4. 告诉 DeepSeek 能使用什么工具 ==========

tool_definition = {
    "type": "function",
    "function": {
        "name": "evaluate_localization_strategy",
        "description": (
            "对十节点鲁棒定位策略进行可复现的模拟实验。"
            "比较定位平均误差、CEP90、不同噪声场景的"
            "表现以及异常节点识别率。"
            "仅返回聚合实验指标，不返回单次真实位置。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": [
                        "huber",
                        "fixed_geometry",
                        "adaptive_geometry"
                    ],
                    "description": (
                        "选择定位方法：Huber、"
                        "固定几何阈值或自适应几何阈值"
                    )
                },
                "parameter": {
                    "type": "number",
                    "description": (
                        "Huber 对应 delta；"
                        "固定几何对应 threshold；"
                        "自适应几何对应系数 c"
                    )
                }
            },
            "required": [
                "method",
                "parameter"
            ]
        }
    }
}


# ========== 5. 初始化研究智能体 ==========

messages = [
    {
        "role": "system",
        "content": (
            "你是一名研究鲁棒非线性众包定位的"
            "数学研究智能体。"
            "你可以自主选择三种定位策略及其参数，"
            "并调用 Python 实验工具验证。"
            "当前实验有十个观测节点，"
            "包含不同正常噪声和异常测距幅度。"
            "你的目标是在不同场景下降低 macro_cep90，"
            "同时关注平均误差、异常识别率和极端风险。"
            "每轮根据之前的反馈选择新方案。"
            "不得重复完全相同的方法和参数组合。"
            "尽量比较不同方法，而不是只测试一种方法。"
            "你不能要求查看单次实验的真实目标位置。"
            "不要把模拟实验成功等同于真实部署成功。"
            "不要宣称已解决华为原始难题。"
        )
    },
    {
        "role": "user",
        "content": (
            "请开展三轮十节点鲁棒定位研究。"
            "你可以选择 huber、fixed_geometry、"
            "adaptive_geometry 三种方法。"
            "每轮自主选择方法及参数，"
            "调用工具获取实验结果，"
            "根据反馈继续改进。"
            "请尽量探索不同的策略。"
        )
    }
]


# ========== 6. 运行三轮智能体实验 ==========

history = []

for round_index in range(3):

    print(
        f"\n========== 第 {round_index + 1} 轮 ==========",
        flush=True
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=[tool_definition],
        tool_choice={
            "type": "function",
            "function": {
                "name": "evaluate_localization_strategy"
            }
        },
        temperature=0
    )

    assistant_message = response.choices[0].message

    messages.append(
        assistant_message.model_dump(
            exclude_none=True
        )
    )

    if not assistant_message.tool_calls:
        raise RuntimeError(
            "DeepSeek 本轮没有调用实验工具"
        )

    accepted = False

    for call in assistant_message.tool_calls:

        try:
            args = json.loads(
                call.function.arguments
            )

            method = str(args["method"])
            parameter = float(args["parameter"])

            if accepted:

                result = {
                    "error": "本轮仅允许执行一次实验"
                }

            elif any(
                h["method"] == method
                and abs(
                    h["parameter"] - parameter
                ) < 1e-9
                for h in history
            ):

                result = {
                    "error": "该方法与参数组合已测试"
                }

            else:

                result = evaluate_strategy(
                    method,
                    parameter,
                    development_data
                )

                result["round"] = round_index + 1

                history.append(result)
                accepted = True

                print(
                    "DeepSeek 选择：",
                    method,
                    "参数：",
                    parameter,
                    flush=True
                )

                print(
                    "实验结果：",
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        indent=2
                    ),
                    flush=True
                )

        except (
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            RuntimeError
        ) as exc:

            result = {"error": str(exc)}

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(
                result,
                ensure_ascii=False
            )
        })

    if not accepted:
        raise RuntimeError(
            "本轮未能执行有效实验，请检查工具参数"
        )


# ========== 7. DeepSeek 总结研究结果 ==========

messages.append({
    "role": "user",
    "content": (
        "三轮实验已经完成。"
        "请比较各方法的 macro_cep90、平均误差、"
        "不同噪声下的 CEP90 和异常识别率。"
        "指出当前最好候选策略，并解释原因。"
        "明确说明实验样本有限、"
        "模拟数据与真实场景存在差异。"
        "不要再调用实验工具。"
    )
})

final_response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    temperature=0
)

summary = final_response.choices[0].message.content

print("\n========== DeepSeek 研究总结 ==========")
print(summary)


# ========== 8. 独立测试当前最好方案 ==========

best = min(
    history,
    key=lambda item: item["macro_cep90"]
)

print("\n========== 独立随机测试 ==========")
print("开发集表现最好的方案：")
print(best["method"], best["parameter"])

holdout_result = evaluate_strategy(
    best["method"],
    best["parameter"],
    holdout_data
)

print(
    json.dumps(
        holdout_result,
        ensure_ascii=False,
        indent=2
    )
)


# ========== 9. 保存完整研究记录 ==========

record = {
    "agent_model": "deepseek-chat",
    "development_seed": 2033,
    "holdout_seed": 3033,
    "experiments": history,
    "agent_summary": summary,
    "best_development_strategy": best,
    "holdout_result": holdout_result
}

with open(
    "localization_agent_v2_history.json",
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
    "\n研究记录已保存："
    "localization_agent_v2_history.json"
)
