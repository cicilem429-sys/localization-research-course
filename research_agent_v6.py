
import os
import json
from pathlib import Path
from itertools import combinations

import numpy as np

from research_environment_v3 import (
    make_datasets,
    run_strategy
)
from weighted_localization import (
    weighted_huber_localization
)


# ==========================================
# 1. 研究配置
# ==========================================


MODEL = "deepseek-v4-pro"
ROUNDS = 16

# 之前 V1-V5 累计估计费用
PREVIOUS_SPEND_CNY = 0.479

TOTAL_BUDGET_CNY = 100.0
SAFETY_RESERVE_CNY = 5.0

# 保守估算参数，不代表实时官方账单
USD_TO_CNY = 7.5
INPUT_PRICE = 1.32
OUTPUT_PRICE = 3.96

HISTORY_FILE = Path(
    "research_agent_v6_history.json"
)

client = None


def get_client():
    """Initialize the API client only for an explicitly started agent run."""
    global client
    if client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=180.0,
            max_retries=1
        )
    return client

spent = 0.0
input_tokens = 0
output_tokens = 0
request_count = 0

history = []


# ==========================================
# 2. API 调用和费用记录
# ==========================================

def ask_model(messages, tools):

    global spent
    global input_tokens
    global output_tokens
    global request_count

    remaining = (
        TOTAL_BUDGET_CNY
        - PREVIOUS_SPEND_CNY
        - spent
        - SAFETY_RESERVE_CNY
    )

    if remaining < 5.0:
        raise RuntimeError(
            "API 预算不足，停止研究"
        )

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {
                "name": "test_geometry_support"
            }
        },
        max_tokens=1600,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )

    usage = response.usage

    if usage is not None:

        n_in = usage.prompt_tokens or 0
        n_out = usage.completion_tokens or 0

        estimated_cost = (
            n_in * INPUT_PRICE
            + n_out * OUTPUT_PRICE
        ) / 1_000_000 * USD_TO_CNY

        spent += estimated_cost
        input_tokens += n_in
        output_tokens += n_out

        print(
            f"本轮预计费用：¥{estimated_cost:.4f}",
            flush=True
        )
        print(
            f"V6累计预计费用：¥{spent:.4f}",
            flush=True
        )

    request_count += 1

    return response.choices[0].message


# ==========================================
# 3. 构造三节点几何候选位置
# ==========================================

def generate_candidates(anchors, distances):

    anchors = np.asarray(
        anchors, dtype=float
    )

    distances = np.asarray(
        distances, dtype=float
    )

    candidates = []
    origins = []

    for ids in combinations(
        range(len(anchors)), 3
    ):

        i, j, k = ids

        p0 = anchors[i]
        p1 = anchors[j]
        p2 = anchors[k]

        # 从平方测距方程相减得到线性系统
        A = 2.0 * np.vstack([
            p1 - p0,
            p2 - p0
        ])

        b = np.array([
            distances[i] ** 2
            - distances[j] ** 2
            + np.dot(p1, p1)
            - np.dot(p0, p0),

            distances[i] ** 2
            - distances[k] ** 2
            + np.dot(p2, p2)
            - np.dot(p0, p0)
        ])

        # 排除几何退化或病态三角形
        if np.linalg.cond(A) > 1e5:
            continue

        try:
            position = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue

        if np.all(np.isfinite(position)):
            candidates.append(position)
            origins.append(ids)

    if not candidates:
        raise RuntimeError(
            "未能生成有效几何候选位置"
        )

    return (
        np.asarray(candidates),
        np.asarray(origins, dtype=int)
    )


# ==========================================
# 4. 预计算观测数据和候选残差
# ==========================================

def prepare_data(datasets):

    prepared = []

    for index, item in enumerate(datasets):

        anchors = np.asarray(
            item["anchors"], dtype=float
        )

        distances = np.asarray(
            item["distances"], dtype=float
        )

        candidates, origins = generate_candidates(
            anchors, distances
        )

        candidate_residuals = np.abs(
            np.linalg.norm(
                candidates[:, None, :]
                - anchors[None, :, :],
                axis=2
            ) - distances[None, :]
        )

        fixed = run_strategy(
            "fixed_geometry", 1.5,
            anchors, distances
        )

        adaptive = run_strategy(
            "adaptive_geometry", 2.0,
            anchors, distances
        )

        huber = run_strategy(
            "huber", 1.0,
            anchors, distances
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
            "candidates": candidates,
            "origins": origins,
            "candidate_residuals": candidate_residuals,
            "fixed": np.asarray(
                fixed["position"], dtype=float
            ),
            "adaptive": np.asarray(
                adaptive["position"], dtype=float
            ),
            "huber": np.asarray(
                huber["position"], dtype=float
            )
        })

        if (index + 1) % 40 == 0:
            print(
                f"已准备 {index + 1}"
                f"/{len(datasets)} 组",
                flush=True
            )

    return prepared


# ==========================================
# 5. 参数验证
# ==========================================

def validate(p):

    result = {
        "threshold": float(p["threshold"]),
        "top_k": int(p["top_k"]),
        "gamma": float(p["gamma"]),
        "min_weight": float(p["min_weight"]),
        "delta": float(p["delta"]),
        "temperature": float(p["temperature"])
    }

    if not all(
        np.isfinite(v)
        for v in result.values()
    ):
        raise ValueError("存在无效参数")

    if not 0.2 <= result["threshold"] <= 3.0:
        raise ValueError("threshold 超出范围")

    if not 3 <= result["top_k"] <= 120:
        raise ValueError("top_k 超出范围")

    if not 0.3 <= result["gamma"] <= 4.0:
        raise ValueError("gamma 超出范围")

    if not 0 <= result["min_weight"] <= 0.5:
        raise ValueError("min_weight 超出范围")

    if not 0.1 <= result["delta"] <= 3.0:
        raise ValueError("delta 超出范围")

    if not 0.2 <= result["temperature"] <= 5.0:
        raise ValueError("temperature 超出范围")

    return result


# ==========================================
# 6. 多候选几何支持度权重
# ==========================================

def geometry_support_weights(item, p):

    threshold = p["threshold"]
    residuals = item["candidate_residuals"]
    origins = item["origins"]

    # 候选的几何一致性得分
    consistent = residuals <= threshold

    # 不用构造该候选的三个节点给其自身评分
    eligible = np.ones_like(
        consistent, dtype=bool
    )

    rows = np.arange(len(origins))[:, None]

    eligible[rows, origins] = False

    scores = np.sum(
        consistent & eligible,
        axis=1
    )

    # 优先使用一致性得分高的候选解
    k = min(p["top_k"], len(scores))

    order = np.argsort(
        -scores, kind="stable"
    )[:k]

    selected_scores = scores[order]

    # 使用softmax进行候选质量加权
    scaled = (
        selected_scores
        - np.max(selected_scores)
    ) / p["temperature"]

    quality = np.exp(scaled)
    quality = quality / np.sum(quality)

    # 计算每个节点得到的支持比例
    # 只用不包含该节点的候选给它投票
    selected_consistent = consistent[order]
    selected_eligible = eligible[order]

    weighted_votes = (
        quality[:, None]
        * selected_consistent
        * selected_eligible
    )

    denominators = np.sum(
        quality[:, None] * selected_eligible,
        axis=0
    )

    numerators = np.sum(
        weighted_votes,
        axis=0
    )

    support = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators, dtype=float),
        where=denominators > 1e-12
    )

    # 当某个节点没有独立投票时，
    # 使用所有候选对它的支持作为回退。
    missing = denominators <= 1e-12

    if np.any(missing):
        support[missing] = (
            quality
            @ selected_consistent[:, missing].astype(float)
        )

    # 支持度映射为连续可信权重
    weights = (
        p["min_weight"]
        + (1.0 - p["min_weight"])
        * support ** p["gamma"]
    )

    return (
        weights,
        support,
        int(k)
    )


# ==========================================
# 7. 评价指标
# ==========================================

def summarize(data, errors):

    errors = np.asarray(errors, dtype=float)

    if not np.all(np.isfinite(errors)):
        raise ValueError("存在无效定位误差")

    grouped = {}

    for item, error in zip(data, errors):

        key = (
            item["scenario"],
            item["sigma"]
        )

        grouped.setdefault(
            key, []
        ).append(float(error))

    case_scores = []
    per_case = {}

    for (scenario, sigma), values in grouped.items():

        score = float(
            np.percentile(values, 90)
        )

        case_scores.append(score)

        per_case[
            f"{scenario}|sigma={sigma}"
        ] = round(score, 4)

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


def evaluate_baseline(data, method):

    errors = [
        float(np.linalg.norm(
            item[method] - item["true_position"]
        ))
        for item in data
    ]

    return summarize(data, errors)


# V5最佳候选作为额外对照
V5_PROPOSAL = {
    "tau": 4.0,
    "delta": 1.0
}


def evaluate_v5(data):

    errors = []

    for item in data:

        initial = item["fixed"]

        residuals = (
            np.linalg.norm(
                item["anchors"] - initial,
                axis=1
            )
            - item["distances"]
        )

        weights = np.exp(
            -0.5 * (
                np.abs(residuals)
                / V5_PROPOSAL["tau"]
            ) ** 2
        )

        result = weighted_huber_localization(
            item["anchors"],
            item["distances"],
            weights,
            delta=V5_PROPOSAL["delta"],
            initial_position=initial
        )

        error = np.linalg.norm(
            np.asarray(result["position"])
            - item["true_position"]
        )

        errors.append(float(error))

    return summarize(data, errors)


# ==========================================
# 8. 评价V6候选算法
# ==========================================

def evaluate_candidate(data, proposal):

    p = validate(proposal)

    errors = []
    average_weights = []
    average_supports = []

    for item in data:

        weights, support, _ = (
            geometry_support_weights(item, p)
        )

        result = weighted_huber_localization(
            item["anchors"],
            item["distances"],
            weights,
            delta=p["delta"],
            initial_position=item["adaptive"]
        )

        error = float(np.linalg.norm(
            np.asarray(result["position"])
            - item["true_position"]
        ))

        if not np.isfinite(error):
            raise RuntimeError(
                "候选算法出现无效定位结果"
            )

        errors.append(error)
        average_weights.append(
            float(np.mean(weights))
        )
        average_supports.append(
            float(np.mean(support))
        )

    metrics = summarize(data, errors)

    metrics["average_weight"] = round(
        float(np.mean(average_weights)), 4
    )

    metrics["average_support"] = round(
        float(np.mean(average_supports)), 4
    )

    return metrics


# ==========================================
# 9. 保存结果
# ==========================================

def save_history(
    development_baselines=None,
    final=None,
    error=None
):

    record = {
        "version": "V6",
        "model": MODEL,
        "round_limit": ROUNDS,
        "development_seed": 2063,
        "holdout_seed": 9063,
        "experiments": history,
        "api": {
            "request_count": request_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_v6_cost_cny": round(
                spent, 6
            ),
            "previous_spend_cny": PREVIOUS_SPEND_CNY
        }
    }

    if development_baselines is not None:
        record["development_baselines"] = (
            development_baselines
        )

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
# 10. DeepSeek研究工具
# ==========================================

research_tool = {
    "type": "function",
    "function": {
        "name": "test_geometry_support",
        "description": (
            "测试基于多候选几何一致性支持度的"
            "加权Huber二维定位算法。"
            "返回12个场景的定位精度。"
            "每轮先给出数学假设。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": (
                        "几何一致性残差阈值，0.2到3"
                    )
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "使用排名前多少个候选，3到120"
                    )
                },
                "gamma": {
                    "type": "number",
                    "description": (
                        "支持度幂指数，0.3到4"
                    )
                },
                "min_weight": {
                    "type": "number",
                    "description": (
                        "最低权重，0到0.5"
                    )
                },
                "delta": {
                    "type": "number",
                    "description": (
                        "Huber损失阈值，0.1到3"
                    )
                },
                "temperature": {
                    "type": "number",
                    "description": (
                        "候选质量softmax温度，0.2到5"
                    )
                },
                "hypothesis": {
                    "type": "string",
                    "description": (
                        "本轮数学假设，"
                        "说明为什么选择这些参数，"
                        "以及上一轮失败的原因"
                    )
                }
            },
            "required": [
                "threshold",
                "top_k",
                "gamma",
                "min_weight",
                "delta",
                "temperature",
                "hypothesis"
            ]
        }
    }
}


# ==========================================
# 11. 主程序
# ==========================================

def main():

    print(
        "===== V6 几何支持度数学研究智能体 =====",
        flush=True
    )

    print("模型：", MODEL)
    print("研究轮数：", ROUNDS)

    print("\n构建开发集...", flush=True)

    dev = prepare_data(
        make_datasets(
            seed=2063,
            trials_per_case=10
        )
    )

    print("\n构建独立测试集...", flush=True)

    holdout = prepare_data(
        make_datasets(
            seed=9063,
            trials_per_case=20
        )
    )

    baselines = {
        method: evaluate_baseline(dev, method)
        for method in (
            "huber",
            "fixed",
            "adaptive"
        )
    }

    print("\n计算 V5 方法基线...", flush=True)

    baselines["v5"] = evaluate_v5(dev)

    print("\n===== 开发集基线 =====")

    for name, metrics in baselines.items():

        print(
            name,
            "mean_error =",
            metrics["mean_error"],
            "macro_cep90 =",
            metrics["macro_cep90"]
        )

    messages = [
        {
            "role": "system",
            "content": (
                "你是研究二维众包鲁棒定位的数学智能体。"
                "V5探索了残差加权，但未超过自适应几何基线。"
                "V6研究多候选几何支持度加权。"
                "候选由三节点测距方程生成，"
                "候选质量根据其他节点的一致性计算。"
                "节点权重为"
                "w=min_weight+(1-min_weight)*support^gamma。"
                "候选按一致性得分排序，"
                "使用top_k个候选并通过softmax加权。"
                "你可以调整阈值、top_k、gamma、"
                "最低权重、Huber delta和softmax温度。"
                "每轮必须提出具体数学假设，"
                "调用工具评价，然后根据反馈改变方案。"
                "优化目标是macro_cep90越小越好。"
                "不能编造结果，不能重复参数。"
                "任何改进必须和所有基线比较。"
                "不得声称现有方法的组合是原创理论。"
                "不能访问真实位置或真实异常标签。"
            )
        },
        {
            "role": "user",
            "content": (
                "请开始16轮数学研究。"
                "每轮结合上一轮的实验反馈提出假设，"
                "探索多候选几何一致性支持度的权重规则。"
                "开发集基线macro_cep90："
                + json.dumps(
                    {
                        name: metrics["macro_cep90"]
                        for name, metrics in baselines.items()
                    },
                    ensure_ascii=False
                )
            )
        }
    ]

    seen = set()

    for round_number in range(
        1, ROUNDS + 1
    ):

        print(
            f"\n========== V6第{round_number}轮 ==========",
            flush=True
        )

        try:
            message = ask_model(
                messages,
                [research_tool]
            )
        except Exception as exc:
            save_history(
                development_baselines=baselines,
                error=exc
            )
            print("API请求失败：", exc)
            break

        messages.append(
            message.model_dump(
                exclude_none=True
            )
        )

        calls = message.tool_calls or []

        if not calls:
            print("模型没有调用实验工具")
            break

        valid_round = False

        for call in calls:

            try:
                if valid_round:
                    raise ValueError(
                        "每轮最多一个有效实验"
                    )

                args = json.loads(
                    call.function.arguments
                )

                proposal = validate(args)

                signature = tuple(
                    proposal[key]
                    for key in (
                        "threshold",
                        "top_k",
                        "gamma",
                        "min_weight",
                        "delta",
                        "temperature"
                    )
                )

                if signature in seen:
                    raise ValueError(
                        "重复的候选参数"
                    )

                seen.add(signature)

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
                    "研究假设：",
                    entry["hypothesis"],
                    flush=True
                )

                print(
                    "候选参数：",
                    proposal,
                    flush=True
                )

                print(
                    "平均定位误差：",
                    metrics["mean_error"]
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
                    "工具错误：",
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

        save_history(
            development_baselines=baselines
        )

        if not valid_round:
            print("本轮未成功，结束研究循环")
            break

        best = min(
            history,
            key=lambda item:
                item["metrics"]["macro_cep90"]
        )

        print(
            "目前最佳 macro-CEP90：",
            best["metrics"]["macro_cep90"]
        )

    # ======================================
    # 12. 选择最佳候选
    # ======================================

    if not history:
        print("没有有效候选方案")
        return

    best = min(
        history,
        key=lambda item:
            item["metrics"]["macro_cep90"]
    )

    print("\n===== V6开发集最佳方案 =====")
    print(best["proposal"])
    print(
        "macro-CEP90：",
        best["metrics"]["macro_cep90"]
    )

    # ======================================
    # 13. 测试集验证
    # ======================================

    print("\n===== V6独立测试集验证 =====")

    test_baselines = {
        method: evaluate_baseline(
            holdout, method
        )
        for method in (
            "huber",
            "fixed",
            "adaptive"
        )
    }

    test_baselines["v5"] = evaluate_v5(
        holdout
    )

    test_candidate = evaluate_candidate(
        holdout,
        best["proposal"]
    )

    for name, metrics in test_baselines.items():

        print(
            name,
            "macro-CEP90：",
            metrics["macro_cep90"]
        )

    print(
        "V6候选 macro-CEP90：",
        test_candidate["macro_cep90"]
    )

    baseline_name, baseline_metrics = min(
        test_baselines.items(),
        key=lambda pair:
            pair[1]["macro_cep90"]
    )

    difference = (
        baseline_metrics["macro_cep90"]
        - test_candidate["macro_cep90"]
    )

    print(
        "最佳基线：",
        baseline_name
    )

    print(
        "基线分数 - V6分数：",
        round(difference, 4)
    )

    if difference > 0:
        print("V6候选在本次测试中领先")
    else:
        print("V6候选未超过最佳基线")

    final = {
        "best_candidate": best,
        "test_baselines": test_baselines,
        "test_candidate": test_candidate,
        "difference": round(difference, 4)
    }

    save_history(
        development_baselines=baselines,
        final=final
    )

    print("\n===== V6研究完成 =====")
    print("实验记录：", HISTORY_FILE)
    print("API调用次数：", request_count)
    print("输入Token：", input_tokens)
    print("输出Token：", output_tokens)
    print(
        "V6预计费用：",
        round(spent, 4),
        "元"
    )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        save_history(error="用户中断")
        print("程序中断，研究记录已保存")

    except Exception as exc:
        save_history(error=exc)
        print(
            "程序发生错误：",
            type(exc).__name__,
            exc
        )
        raise
