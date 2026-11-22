
import os
import json
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

from research_environment_v3 import (
    make_datasets,
    run_strategy
)
from localization_tools import estimate_position_multistart


# ===== 1. 连接 DeepSeek =====

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    timeout=120.0
)


# ===== 2. 构造开发集与独立测试集 =====

development_data = make_datasets(
    seed=2050,
    trials_per_case=10
)

holdout_data = make_datasets(
    seed=9050,
    trials_per_case=10
)


# ===== 3. 计算单次观测的噪声指标 =====

def estimate_noise_feature(anchors, distances):

    huber = estimate_position_multistart(
        anchors,
        distances,
        method="huber",
        delta=0.3
    )

    position = np.asarray(huber["position"])

    residuals = (
        np.linalg.norm(anchors - position, axis=1)
        - distances
    )

    center = np.median(residuals)

    mad = np.median(
        np.abs(residuals - center)
    )

    return float(1.4826 * mad)


# ===== 4. 预先计算候选策略 =====
# 避免每一轮重复进行相同的数值优化

def prepare_data(datasets):
    prepared = []

    for item in datasets:

        anchors = item["anchors"]
        distances = item["distances"]

        sigma_hat = estimate_noise_feature(
            anchors, distances
        )

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

        prepared.append({
            "scenario": item["scenario"],
            "sigma": item["sigma"],
            "true_position": item["true_position"],
            "sigma_hat": sigma_hat,
            "fixed_position": fixed["position"],
            "adaptive_position": adaptive["position"]
        })

    return prepared


print("正在准备开发集实验...", flush=True)
dev = prepare_data(development_data)

print("正在准备保留集实验...", flush=True)
holdout = prepare_data(holdout_data)

print("开发集样本：", len(dev), flush=True)
print("保留集样本：", len(holdout), flush=True)


# ===== 5. 评价策略切换规则 =====

def evaluate_switch(data, boundary, direction):

    boundary = float(boundary)

    if not np.isfinite(boundary):
        raise ValueError("切换边界必须是有限数值")

    if not 0.05 <= boundary <= 2.0:
        raise ValueError("boundary 范围为 0.05 到 2.0")

    if direction not in ("low_fixed", "high_fixed"):
        raise ValueError("不支持的切换方向")

    grouped = {}
    all_errors = []
    fixed_count = 0

    for item in data:

        low_noise = item["sigma_hat"] <= boundary

        if direction == "low_fixed":
            use_fixed = low_noise
        else:
            use_fixed = not low_noise

        if use_fixed:
            position = item["fixed_position"]
            fixed_count += 1
        else:
            position = item["adaptive_position"]

        error = float(np.linalg.norm(
            np.asarray(position)
            - item["true_position"]
        ))

        if not np.isfinite(error):
            raise RuntimeError("定位误差无效")

        key = (
            item["scenario"],
            item["sigma"]
        )

        grouped.setdefault(key, []).append(error)
        all_errors.append(error)

    per_case = {}
    case_scores = []

    for (scenario, sigma), errors in grouped.items():

        score = float(np.percentile(errors, 90))
        case_scores.append(score)

        per_case[f"{scenario}|sigma={sigma}"] = {
            "mean_error": round(float(np.mean(errors)), 4),
            "cep90": round(score, 4)
        }

    return {
        "boundary": boundary,
        "direction": direction,
        "mean_error": round(
            float(np.mean(all_errors)), 4
        ),
        "cep90": round(
            float(np.percentile(all_errors, 90)), 4
        ),
        "macro_cep90": round(
            float(np.mean(case_scores)), 4
        ),
        "max_error": round(
            float(np.max(all_errors)), 4
        ),
        "fixed_selection_rate": round(
            fixed_count / len(data), 4
        ),
        "per_case": per_case
    }


# ===== 6. 计算不切换时的基线 =====

def evaluate_baseline(data, method):

    errors = []
    grouped = {}

    for item in data:

        position = item[f"{method}_position"]

        error = float(np.linalg.norm(
            np.asarray(position) - item["true_position"]
        ))

        errors.append(error)

        key = (item["scenario"], item["sigma"])
        grouped.setdefault(key, []).append(error)

    return {
        "method": method,
        "mean_error": round(float(np.mean(errors)), 4),
        "macro_cep90": round(float(np.mean([
            np.percentile(v, 90)
            for v in grouped.values()
        ])), 4),
        "cep90": round(float(np.percentile(errors, 90)), 4)
    }


dev_fixed = evaluate_baseline(dev, "fixed")
dev_adaptive = evaluate_baseline(dev, "adaptive")

print("\n===== 开发集基线 =====")
print("固定几何：", dev_fixed)
print("自适应几何：", dev_adaptive)


# ===== 7. 告诉 DeepSeek 可调用什么工具 =====

tool_definition = {
    "type": "function",
    "function": {
        "name": "test_switch_rule",
        "description": (
            "测试基于观测噪声指标的两种定位策略切换规则。"
            "返回多场景平均误差和 macro_cep90。"
            "真实位置只用于实验评分。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "boundary": {
                    "type": "number",
                    "description": (
                        "切换边界，取值 0.05 到 2.0"
                    )
                },
                "direction": {
                    "type": "string",
                    "enum": [
                        "low_fixed",
                        "high_fixed"
                    ],
                    "description": (
                        "low_fixed：低噪声估计时用固定几何；"
                        "high_fixed：高噪声估计时用固定几何"
                    )
                }
            },
            "required": [
                "boundary",
                "direction"
            ]
        }
    }
}


# ===== 8. 初始化研究智能体 =====

messages = [
    {
        "role": "system",
        "content": (
            "你是研究二维鲁棒众包定位的数学智能体。"
            "你正在研究基于观测残差 MAD 的算法切换规则。"
            "固定几何方法阈值为1.5，"
            "自适应几何方法系数为2.0。"
            "你能控制切换边界和切换方向。"
            "请根据上一轮实验反馈提出下一轮方案。"
            "核心评价指标是 macro_cep90，越小越好。"
            "必须与不切换的基线比较。"
            "不能读取单次真实位置或真实噪声水平。"
            "不能重复完全相同的规则。"
            "不得把已有方法的组合宣称为原创理论。"
            "如果切换方法没有改善，应当如实说明。"
        )
    },
    {
        "role": "user",
        "content": (
            "请进行四轮数学实验。"
            "每轮通过工具提出一个切换规则。"
            "结合上一轮实验反馈修正研究假设。"
            "目标是改善跨场景定位性能。"
            "开发集固定几何基线："
            + json.dumps(dev_fixed, ensure_ascii=False)
            + "；自适应几何基线："
            + json.dumps(dev_adaptive, ensure_ascii=False)
        )
    }
]


# ===== 9. 四轮 DeepSeek 实验 =====

history = []

for round_index in range(4):

    print(
        f"\n========== V4 第 {round_index + 1} 轮 ==========",
        flush=True
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=[tool_definition],
        tool_choice={
            "type": "function",
            "function": {
                "name": "test_switch_rule"
            }
        },
        temperature=0
    )

    assistant_message = response.choices[0].message

    messages.append(
        assistant_message.model_dump(exclude_none=True)
    )

    if not assistant_message.tool_calls:
        raise RuntimeError("DeepSeek 没有调用工具")

    accepted = False

    for call in assistant_message.tool_calls:

        try:
            args = json.loads(call.function.arguments)

            boundary = float(args["boundary"])
            direction = str(args["direction"])

            if accepted:
                result = {
                    "error": "本轮只允许一次有效实验"
                }

            elif any(
                h["direction"] == direction
                and abs(h["boundary"] - boundary) < 1e-9
                for h in history
            ):
                result = {
                    "error": "该规则已经测试过"
                }

            else:
                result = evaluate_switch(
                    dev, boundary, direction
                )

                result["round"] = round_index + 1
                history.append(result)
                accepted = True

                print(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        indent=2
                    ),
                    flush=True
                )

        except (
            ValueError,
            TypeError,
            KeyError,
            RuntimeError
        ) as exc:
            result = {"error": str(exc)}

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(
                result, ensure_ascii=False
            )
        })

    # 每轮保存一次
    with open(
        "research_agent_v4_history.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "development_baselines": [
                    dev_fixed, dev_adaptive
                ],
                "experiments": history
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    if not accepted:
        raise RuntimeError("本轮没有执行有效实验")


# ===== 10. 总结并选择候选 =====

messages.append({
    "role": "user",
    "content": (
        "实验结束，请不要再调用工具。"
        "比较四轮切换规则和两种原始基线。"
        "分析切换是否带来了真实改进。"
        "若没有改善，必须明确指出。"
        "解释失败场景与当前研究局限。"
    )
})

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    temperature=0
)

summary = response.choices[0].message.content

print("\n========== DeepSeek V4 研究总结 ==========")
print(summary)

best = min(
    history,
    key=lambda x: x["macro_cep90"]
)

print("\n========== 开发集最佳切换规则 ==========")
print(best["direction"], best["boundary"])
print("macro-CEP90：", best["macro_cep90"])


# ===== 11. 保留集独立验证 =====

print("\n========== 保留集公平对照 ==========")

holdout_fixed = evaluate_baseline(holdout, "fixed")
holdout_adaptive = evaluate_baseline(holdout, "adaptive")

holdout_switch = evaluate_switch(
    holdout,
    best["boundary"],
    best["direction"]
)

print("固定几何：", holdout_fixed)
print("自适应几何：", holdout_adaptive)
print("智能体切换：", holdout_switch)


# ===== 12. 保存最终结果 =====

record = {
    "model": "deepseek-chat",
    "development_seed": 2050,
    "holdout_seed": 9050,
    "development_baselines": [
        dev_fixed, dev_adaptive
    ],
    "experiments": history,
    "agent_summary": summary,
    "best_rule": best,
    "holdout_baselines": [
        holdout_fixed, holdout_adaptive
    ],
    "holdout_switch": holdout_switch
}

with open(
    "research_agent_v4_history.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(record, f, ensure_ascii=False, indent=2)

print("\nV4 研究记录已保存")
