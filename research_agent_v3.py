
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from research_environment_v3 import (
    make_datasets,
    evaluate_strategy
)


# ========== 1. 连接 DeepSeek ==========

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "请检查 .env 中的 DEEPSEEK_API_KEY"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    timeout=120.0
)


# ========== 2. 准备实验数据 ==========

# 开发集：允许智能体反复比较
development_data = make_datasets(
    seed=2041,
    trials_per_case=10
)

# 保留集：只评价最后选出的方案
holdout_data = make_datasets(
    seed=9041,
    trials_per_case=10
)

print("开发集样本数：", len(development_data))
print("保留集样本数：", len(holdout_data))


# ========== 3. 定义实验工具 ==========

tool_definition = {
    "type": "function",
    "function": {
        "name": "evaluate_localization",
        "description": (
            "运行十二种场景组合的鲁棒定位实验。"
            "返回平均误差、CEP90、macro_cep90、"
            "最大误差、异常检测率和各场景指标。"
            "真实位置只用于模拟实验评价，"
            "不提供给定位算法。"
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
                    "description": "定位方法"
                },
                "parameter": {
                    "type": "number",
                    "description": (
                        "Huber: delta 范围 0.1~4；"
                        "固定几何: threshold 范围 0.3~4；"
                        "自适应几何: c 范围 0.5~5"
                    )
                }
            },
            "required": ["method", "parameter"]
        }
    }
}


# ========== 4. 初始化智能体 ==========

messages = [
    {
        "role": "system",
        "content": (
            "你是一名研究众包鲁棒定位的数学实验智能体。"
            "你的任务是利用真实的 Python 实验工具，"
            "探索不同定位方法及参数。"
            "当前实验环境包括标准场景、随机节点布局、"
            "两个异常节点、小幅异常观测，"
            "并考虑三种正常噪声强度。"
            "目标是降低 macro_cep90，"
            "同时关注平均误差、最大误差和异常检测率。"
            "请根据上一轮的实际实验反馈调整下一轮方案。"
            "每轮必须调用实验工具。"
            "不要重复已经测试的方法和参数组合。"
            "你只能选择已有的三种数学方法，"
            "不能声称自己已经创造了全新算法。"
            "不能读取保留集，也不能获取单次真实位置。"
            "请客观分析失败场景及方法局限。"
        )
    },
    {
        "role": "user",
        "content": (
            "请进行三轮鲁棒定位数学研究。"
            "每轮自主选择一种方法及参数并调用工具。"
            "请根据实验反馈改进研究选择，"
            "尽可能探索不同策略。"
            "重点关注跨场景的 macro_cep90。"
        )
    }
]


# ========== 5. 保存实验记录 ==========

history = []


def save_record(summary=None, holdout=None):
    record = {
        "model": "deepseek-chat",
        "development_seed": 2041,
        "holdout_seed": 9041,
        "trials_per_case": 10,
        "experiments": history,
        "agent_summary": summary,
        "holdout_result": holdout
    }

    with open(
        "research_agent_v3_history.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=2
        )


# ========== 6. 三轮智能体研究 ==========

for round_index in range(3):

    print(
        f"\n========== V3 第 {round_index + 1} 轮 ==========",
        flush=True
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=[tool_definition],
        tool_choice={
            "type": "function",
            "function": {
                "name": "evaluate_localization"
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
            "DeepSeek 没有调用实验工具"
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
                    "error": "本轮只允许一次有效实验"
                }

            elif any(
                old["method"] == method
                and abs(
                    old["parameter"] - parameter
                ) < 1e-9
                for old in history
            ):
                result = {
                    "error": "该方法和参数组合已经测试"
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
                    "平均误差：",
                    result["mean_error"],
                    flush=True
                )

                print(
                    "macro-CEP90：",
                    result["macro_cep90"],
                    flush=True
                )

                print(
                    "异常识别率：",
                    result["exact_outlier_detection_rate"],
                    flush=True
                )

                print(
                    "各场景结果：",
                    json.dumps(
                        result["per_case"],
                        ensure_ascii=False,
                        indent=2
                    ),
                    flush=True
                )

                # 每轮立即保存，防止中途失败丢失结果
                save_record()

        except (
            ValueError,
            KeyError,
            TypeError,
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
            "本轮未执行有效实验，已停止"
        )


# ========== 7. DeepSeek 生成研究总结 ==========

messages.append({
    "role": "user",
    "content": (
        "三轮实验结束。不要再调用工具。"
        "请比较三种候选策略的实际实验指标，"
        "指出各自适用和失败的场景。"
        "从已测试方案中选出 macro_cep90 最低的候选。"
        "说明三轮探索不足以证明全局最优，"
        "并提出下一阶段的研究建议。"
    )
})

final_response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    temperature=0
)

summary = final_response.choices[0].message.content

print("\n========== DeepSeek V3 研究总结 ==========")
print(summary)

save_record(summary=summary)


# ========== 8. 选择开发集最佳候选 ==========

best = min(
    history,
    key=lambda item: item["macro_cep90"]
)

print("\n========== 开发集最佳候选 ==========")
print("方法：", best["method"])
print("参数：", best["parameter"])
print("macro-CEP90：", best["macro_cep90"])


# ========== 9. 保留集验证 ==========

print("\n========== V3 保留集测试 ==========")

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

# 保存最终记录
save_record(
    summary=summary,
    holdout=holdout_result
)

print(
    "\nV3 研究记录已保存："
    "research_agent_v3_history.json"
)
