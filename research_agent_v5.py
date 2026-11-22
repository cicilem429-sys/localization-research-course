
import os
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from research_environment_v3 import (
    make_datasets,
    run_strategy
)

from weighted_localization import (
    weighted_huber_localization
)


# ==========================================
# 1. 基本配置
# ==========================================

load_dotenv()

MODEL = "deepseek-v4-pro"

# 自动研究轮数
ROUNDS = 12

# 之前已经消耗的 API 费用
PREVIOUS_SPEND_CNY = 0.03

# 总预算和安全预留
TOTAL_BUDGET_CNY = 100.0
SAFETY_RESERVE_CNY = 5.0

# 保守费用估算参数
USD_TO_CNY = 7.5
INPUT_PRICE_USD_PER_M = 1.32
OUTPUT_PRICE_USD_PER_M = 3.96

# 每次请求的保守安全额度
REQUEST_RESERVE_CNY = 3.0

HISTORY_FILE = Path(
    "research_agent_v5_history.json"
)

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    timeout=180.0,
    max_retries=1
)

spent_v5 = 0.0
input_tokens_total = 0
output_tokens_total = 0
request_count = 0


# ==========================================
# 2. API 调用及费用记录
# ==========================================

def check_budget():

    remaining = (
        TOTAL_BUDGET_CNY
        - PREVIOUS_SPEND_CNY
        - spent_v5
        - SAFETY_RESERVE_CNY
    )

    if remaining < REQUEST_RESERVE_CNY:
        raise RuntimeError(
            "API 预计费用已接近预算上限，停止研究。"
        )


def ask_model(messages, tools=None):

    global spent_v5
    global input_tokens_total
    global output_tokens_total
    global request_count

    check_budget()

    kwargs = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 1500,
        "extra_body": {
            "thinking": {
                "type": "disabled"
            }
        }
    }

    if tools is not None:
        kwargs["tools"] = tools

        kwargs["tool_choice"] = {
            "type": "function",
            "function": {
                "name": "evaluate_weight_rule"
            }
        }

    response = client.chat.completions.create(
        **kwargs
    )

    usage = response.usage

    if usage is not None:

        input_tokens = usage.prompt_tokens or 0
        output_tokens = usage.completion_tokens or 0

        cost_usd = (
            input_tokens * INPUT_PRICE_USD_PER_M
            + output_tokens * OUTPUT_PRICE_USD_PER_M
        ) / 1_000_000

        cost_cny = cost_usd * USD_TO_CNY

        spent_v5 += cost_cny
        input_tokens_total += input_tokens
        output_tokens_total += output_tokens

        print(
            f"本次 API 预计费用：¥{cost_cny:.4f}",
            flush=True
        )

        print(
            f"V5 累计预计费用：¥{spent_v5:.4f}",
            flush=True
        )

        print(
            f"Token：输入 {input_tokens}，"
            f"输出 {output_tokens}",
            flush=True
        )

    request_count += 1

    return response.choices[0].message


# ==========================================
# 3. 权重函数
# ==========================================

def make_weights(
    residuals,
    family,
    tau,
    min_weight
):

    r = np.abs(
        np.asarray(residuals, dtype=float)
    )

    if family == "cauchy":

        weights = 1.0 / (
            1.0 + (r / tau) ** 2
        )

    elif family == "gaussian":

        weights = np.exp(
            -0.5 * (r / tau) ** 2
        )

    elif family == "clipped_inverse":

        weights = np.minimum(
            1.0,
            tau / np.maximum(r, 1e-12)
        )

    else:
        raise ValueError(
            "未知权重函数：" + str(family)
        )

    return np.maximum(
        weights,
        min_weight
    )


# ==========================================
# 4. 检查 DeepSeek 提出的参数
# ==========================================

def validate_proposal(args):

    family = str(args["family"])
    tau = float(args["tau"])
    delta = float(args["delta"])
    min_weight = float(args["min_weight"])

    if family not in [
        "cauchy",
        "gaussian",
        "clipped_inverse"
    ]:
        raise ValueError("不支持的权重函数")

    values = [tau, delta, min_weight]

    if not all(np.isfinite(v) for v in values):
        raise ValueError("参数不是有限数值")

    if not 0.1 <= tau <= 4.0:
        raise ValueError("tau 不在允许范围")

    if not 0.1 <= delta <= 3.0:
        raise ValueError("delta 不在允许范围")

    if not 0.0 <= min_weight <= 0.5:
        raise ValueError(
            "min_weight 不在允许范围"
        )

    return {
        "family": family,
        "tau": tau,
        "delta": delta,
        "min_weight": min_weight
    }


# ==========================================
# 5. 准备实验数据
# ==========================================

def prepare_data(datasets):

    prepared = []

    for i, item in enumerate(datasets):

        anchors = np.asarray(
            item["anchors"],
            dtype=float
        )

        distances = np.asarray(
            item["distances"],
            dtype=float
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

        huber = run_strategy(
            "huber",
            1.0,
            anchors,
            distances
        )

        initial = np.asarray(
            fixed["position"],
            dtype=float
        )

        residuals = (
            np.linalg.norm(
                anchors - initial,
                axis=1
            )
            - distances
        )

        prepared.append({
            "anchors": anchors,
            "distances": distances,
            "true_position": np.asarray(
                item["true_position"],
                dtype=float
            ),
            "scenario": item["scenario"],
            "sigma": float(item["sigma"]),
            "initial": initial,
            "residuals": residuals,
            "fixed": np.asarray(
                fixed["position"],
                dtype=float
            ),
            "adaptive": np.asarray(
                adaptive["position"],
                dtype=float
            ),
            "huber": np.asarray(
                huber["position"],
                dtype=float
            )
        })

        if (i + 1) % 40 == 0:
            print(
                f"已准备 {i + 1}/{len(datasets)} 组",
                flush=True
            )

    return prepared


# ==========================================
# 6. 统计定位误差
# ==========================================

def summarize_errors(data, errors):

    errors = np.asarray(
        errors,
        dtype=float
    )

    if not np.all(np.isfinite(errors)):
        raise RuntimeError("定位误差包含无效值")

    grouped = {}

    for item, error in zip(data, errors):

        key = (
            item["scenario"],
            item["sigma"]
        )

        grouped.setdefault(
            key, []
        ).append(float(error))

    per_case = {}
    case_scores = []

    for key, values in grouped.items():

        score = float(
            np.percentile(values, 90)
        )

        name = (
            f"{key[0]}|sigma={key[1]}"
        )

        per_case[name] = round(score, 4)
        case_scores.append(score)

    return {
        "mean_error": round(
            float(np.mean(errors)), 4
        ),
        "cep90": round(
            float(np.percentile(errors, 90)), 4
        ),
        "macro_cep90": round(
            float(np.mean(case_scores)), 4
        ),
        "max_error": round(
            float(np.max(errors)), 4
        ),
        "per_case_cep90": per_case
    }


# ==========================================
# 7. 评价已有定位算法
# ==========================================

def evaluate_baseline(data, method):

    errors = []

    for item in data:

        error = np.linalg.norm(
            item[method]
            - item["true_position"]
        )

        errors.append(float(error))

    return summarize_errors(data, errors)


# ==========================================
# 8. 评价 DeepSeek 候选方法
# ==========================================

def evaluate_candidate(data, proposal):

    p = validate_proposal(proposal)

    errors = []
    mean_weights = []
    low_weight_counts = []

    for item in data:

        weights = make_weights(
            item["residuals"],
            p["family"],
            p["tau"],
            p["min_weight"]
        )

        result = weighted_huber_localization(
            item["anchors"],
            item["distances"],
            weights,
            delta=p["delta"],
            initial_position=item["initial"]
        )

        position = np.asarray(
            result["position"],
            dtype=float
        )

        error = float(
            np.linalg.norm(
                position
                - item["true_position"]
            )
        )

        if not np.isfinite(error):
            raise RuntimeError(
                "优化得到无效定位结果"
            )

        errors.append(error)

        mean_weights.append(
            float(np.mean(weights))
        )

        low_weight_counts.append(
            int(np.sum(weights < 0.1))
        )

    metrics = summarize_errors(
        data,
        errors
    )

    metrics["average_weight"] = round(
        float(np.mean(mean_weights)), 4
    )

    metrics["average_low_weight_nodes"] = round(
        float(np.mean(low_weight_counts)), 4
    )

    return metrics


# ==========================================
# 9. 实验记录
# ==========================================

history = []
messages = []

def save_history(final=None, error=None):

    record = {
        "model": MODEL,
        "round_limit": ROUNDS,
        "development_seed": 2052,
        "holdout_seed": 9052,
        "experiments": history,
        "api_statistics": {
            "request_count": request_count,
            "input_tokens": input_tokens_total,
            "output_tokens": output_tokens_total,
            "estimated_v5_cost_cny": round(
                spent_v5, 6
            ),
            "previous_spend_cny": PREVIOUS_SPEND_CNY
        }
    }

    if final is not None:
        record["final"] = final

    if error is not None:
        record["error"] = str(error)

    HISTORY_FILE.write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


# ==========================================
# 10. DeepSeek 数学实验工具
# ==========================================

research_tool = {
    "type": "function",
    "function": {
        "name": "evaluate_weight_rule",
        "description": (
            "在120组开发数据上评价一组"
            "加权Huber定位参数。"
            "返回平均定位误差、CEP90、"
            "macro-CEP90与分场景结果。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "family": {
                    "type": "string",
                    "enum": [
                        "cauchy",
                        "gaussian",
                        "clipped_inverse"
                    ]
                },
                "tau": {
                    "type": "number",
                    "description": (
                        "权重尺度，0.1到4.0"
                    )
                },
                "delta": {
                    "type": "number",
                    "description": (
                        "Huber损失参数，0.1到3.0"
                    )
                },
                "min_weight": {
                    "type": "number",
                    "description": (
                        "权重下界，0到0.5"
                    )
                },
                "hypothesis": {
                    "type": "string",
                    "description": (
                        "说明本轮数学假设、"
                        "参考的上一轮结果、"
                        "以及预期改进原因"
                    )
                }
            },
            "required": [
                "family",
                "tau",
                "delta",
                "min_weight",
                "hypothesis"
            ]
        }
    }
}


# ==========================================
# 11. 运行数学研究智能体
# ==========================================

def main():

    global messages

    print(
        "===== V5 自动数学研究智能体 =====",
        flush=True
    )

    print("模型：", MODEL)
    print("研究轮数：", ROUNDS)

    development_data = make_datasets(
        seed=2052,
        trials_per_case=10
    )

    holdout_data = make_datasets(
        seed=9052,
        trials_per_case=20
    )

    print("\n准备开发集...", flush=True)
    dev = prepare_data(development_data)

    print("\n准备独立测试集...", flush=True)
    holdout = prepare_data(holdout_data)

    baselines = {
        method: evaluate_baseline(dev, method)
        for method in [
            "huber",
            "fixed",
            "adaptive"
        ]
    }

    print("\n===== 开发集基线 =====")

    for name, result in baselines.items():
        print(
            name,
            "平均误差：",
            result["mean_error"],
            "macro-CEP90：",
            result["macro_cep90"]
        )

    messages = [
        {
            "role": "system",
            "content": (
                "你是一名数学研究智能体。"
                "正在研究二维众包鲁棒定位。"
                "你的任务是探索几何残差驱动的"
                "加权Huber非线性优化。"
                "每轮先提出具体数学假设，"
                "再调用工具进行实验。"
                "下一轮必须分析此前的实验反馈。"
                "优先优化macro_cep90，越低越好。"
                "必须与固定几何、自适应几何和"
                "Huber基线进行比较。"
                "不能重复相同参数组合。"
                "不能编造实验结果。"
                "不能宣称已有权重函数是新理论。"
                "如果效果不佳，必须如实说明。"
                "不要要求访问真实位置或真实异常标签。"
            )
        },
        {
            "role": "user",
            "content": (
                "请开始数学研究。"
                "使用实验工具提出并评价权重规则。"
                "每轮总结上一轮为什么成功或失败。"
                "以下为开发集基线："
                + json.dumps(
                    {
                        k: {
                            "mean_error": v["mean_error"],
                            "macro_cep90": v["macro_cep90"]
                        }
                        for k, v in baselines.items()
                    },
                    ensure_ascii=False
                )
            )
        }
    ]

    seen = set()

    # ======================================
    # 12. 自动研究循环
    # ======================================

    for round_number in range(
        1, ROUNDS + 1
    ):

        print(
            f"\n========== 第 {round_number} 轮 ==========",
            flush=True
        )

        try:
            response = ask_model(
                messages,
                tools=[research_tool]
            )

        except Exception as exc:
            save_history(error=exc)

            print(
                "\nAPI 请求失败：",
                exc
            )
            break

        # 保存模型的工具调用消息
        messages.append(
            response.model_dump(
                exclude_none=True
            )
        )

        tool_calls = response.tool_calls or []

        if not tool_calls:

            print("模型未调用实验工具")
            save_history(
                error="没有工具调用"
            )
            break

        valid_round = False

        for call in tool_calls:

            try:

                if valid_round:
                    raise ValueError(
                        "每轮只能执行一个有效实验"
                    )

                args = json.loads(
                    call.function.arguments
                )

                proposal = validate_proposal(
                    args
                )

                signature = (
                    proposal["family"],
                    proposal["tau"],
                    proposal["delta"],
                    proposal["min_weight"]
                )

                if signature in seen:
                    raise ValueError(
                        "重复的候选参数"
                    )

                seen.add(signature)

                print(
                    "\n数学假设：",
                    args.get(
                        "hypothesis",
                        "未提供"
                    ),
                    flush=True
                )

                print(
                    "候选方案：",
                    proposal,
                    flush=True
                )

                metrics = evaluate_candidate(
                    dev,
                    proposal
                )

                entry = {
                    "round": round_number,
                    "proposal": proposal,
                    "hypothesis": args.get(
                        "hypothesis", ""
                    ),
                    "metrics": metrics
                }

                history.append(entry)
                valid_round = True

                tool_result = metrics

                print(
                    "平均定位误差：",
                    metrics["mean_error"]
                )

                print(
                    "CEP90：",
                    metrics["cep90"]
                )

                print(
                    "macro-CEP90：",
                    metrics["macro_cep90"]
                )

            except Exception as exc:

                tool_result = {
                    "error": str(exc)
                }

                print(
                    "本轮工具错误：",
                    exc
                )

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(
                    tool_result,
                    ensure_ascii=False
                )
            })

        save_history()

        if not valid_round:

            print(
                "本轮没有成功实验，停止研究。"
            )
            break

        best_so_far = min(
            history,
            key=lambda x:
                x["metrics"]["macro_cep90"]
        )

        print(
            "当前最佳 macro-CEP90：",
            best_so_far["metrics"]["macro_cep90"]
        )

    # ======================================
    # 13. 选择开发集最佳候选
    # ======================================

    if not history:
        print(
            "\n没有完成有效研究实验。"
        )
        return

    best = min(
        history,
        key=lambda x:
            x["metrics"]["macro_cep90"]
    )

    print(
        "\n===== V5 开发集最佳方案 ====="
    )

    print(
        "参数：",
        best["proposal"]
    )

    print(
        "开发集 macro-CEP90：",
        best["metrics"]["macro_cep90"]
    )

    # ======================================
    # 14. 独立测试集
    # ======================================

    print(
        "\n===== 独立测试集验证 =====",
        flush=True
    )

    test_baselines = {
        method: evaluate_baseline(
            holdout,
            method
        )
        for method in [
            "huber",
            "fixed",
            "adaptive"
        ]
    }

    test_candidate = evaluate_candidate(
        holdout,
        best["proposal"]
    )

    for name, result in test_baselines.items():

        print(
            name,
            "macro-CEP90：",
            result["macro_cep90"]
        )

    print(
        "V5候选 macro-CEP90：",
        test_candidate["macro_cep90"]
    )

    best_baseline = min(
        test_baselines.items(),
        key=lambda item:
            item[1]["macro_cep90"]
    )

    print(
        "最佳传统基线：",
        best_baseline[0]
    )

    difference = (
        best_baseline[1]["macro_cep90"]
        - test_candidate["macro_cep90"]
    )

    print(
        "基线分数 - V5分数：",
        round(difference, 4)
    )

    if difference > 0:
        print(
            "V5候选本次测试优于最佳基线"
        )

    elif difference < 0:
        print(
            "V5候选本次测试未超过最佳基线"
        )

    else:
        print(
            "V5候选与最佳基线分数相同"
        )

    # ======================================
    # 15. 保存最终研究结果
    # ======================================

    final = {
        "development_baselines": baselines,
        "best_candidate": best,
        "holdout_baselines": test_baselines,
        "holdout_candidate": test_candidate,
        "difference": round(difference, 4)
    }

    save_history(final=final)

    print(
        "\n===== V5 研究完成 ====="
    )

    print(
        "研究记录：",
        HISTORY_FILE
    )

    print(
        "API请求次数：",
        request_count
    )

    print(
        "累计输入Token：",
        input_tokens_total
    )

    print(
        "累计输出Token：",
        output_tokens_total
    )

    print(
        "V5预计费用：",
        round(spent_v5, 4),
        "元"
    )


# ==========================================
# 16. 程序入口
# ==========================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        save_history(
            error="用户手动中断程序"
        )

        print(
            "\n程序已中断，"
            "已完成的实验记录已保存。"
        )

    except Exception as exc:

        save_history(error=exc)

        print(
            "\n程序发生错误：",
            type(exc).__name__,
            str(exc)
        )

        raise
