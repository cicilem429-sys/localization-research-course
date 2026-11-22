
import json
import argparse
import csv
import numpy as np

from pathlib import Path
from scipy.stats import t

from research_environment_v3 import make_datasets

from research_agent_v6 import (
    prepare_data,
    evaluate_baseline,
    evaluate_v5,
    evaluate_candidate
)


# =====================================
# 1. 最终实验设置
# =====================================

SEEDS = [
    11001,
    11002,
    11003,
    11004,
    11005
]

TRIALS_PER_CASE = 10

METHODS = [
    "huber",
    "fixed",
    "adaptive",
    "v5",
    "v6"
]

# V6已经选出的最佳参数，固定不再调整
V6_PROPOSAL = {
    "threshold": 0.5,
    "top_k": 120,
    "gamma": 0.3,
    "min_weight": 0.1,
    "delta": 0.7,
    "temperature": 1.0
}

parser = argparse.ArgumentParser(description="Offline fixed-parameter validation; no API calls")
parser.add_argument("--output-dir", type=Path, default=Path("results/reproduced"))
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = args.output_dir / "final_validation_results.json"
OUTPUT_CSV = args.output_dir / "final_validation_summary.csv"

# =====================================
# 2. 初始化实验记录
# =====================================

results = {
    "experiment": "multi_seed_validation",
    "seeds": SEEDS,
    "trials_per_case": TRIALS_PER_CASE,
    "v6_parameters": V6_PROPOSAL,
    "seed_results": [],
    "summary": {}
}


def save_json():
    OUTPUT_JSON.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


print("===== 最终多随机种子统计验证 =====")
print("随机种子数量：", len(SEEDS))
print("每种子样本数：", TRIALS_PER_CASE * 12)
print("对比算法：", METHODS)


# =====================================
# 3. 对每个随机种子进行公平比较
# =====================================

for seed in SEEDS:

    print(
        f"\n========== 随机种子 {seed} ==========",
        flush=True
    )

    datasets = make_datasets(
        seed=seed,
        trials_per_case=TRIALS_PER_CASE
    )

    data = prepare_data(datasets)

    scores = {}

    for method in (
        "huber",
        "fixed",
        "adaptive"
    ):
        scores[method] = evaluate_baseline(
            data,
            method
        )

    scores["v5"] = evaluate_v5(data)

    scores["v6"] = evaluate_candidate(
        data,
        V6_PROPOSAL
    )

    for method in METHODS:

        metrics = scores[method]

        print(
            f"{method:10s}"
            f" mean={metrics['mean_error']:.4f}"
            f" macroCEP90={metrics['macro_cep90']:.4f}",
            flush=True
        )

    results["seed_results"].append({
        "seed": seed,
        "sample_count": len(data),
        "metrics": scores
    })

    # 每完成一个种子立即保存，避免中断丢失
    save_json()


# =====================================
# 4. 统计每种方法的跨种子表现
# =====================================

print("\n===== 五个随机种子的汇总 =====")

for method in METHODS:

    macro_scores = np.array([
        item["metrics"][method]["macro_cep90"]
        for item in results["seed_results"]
    ])

    mean_errors = np.array([
        item["metrics"][method]["mean_error"]
        for item in results["seed_results"]
    ])

    results["summary"][method] = {
        "macro_cep90_mean": round(
            float(np.mean(macro_scores)), 4
        ),
        "macro_cep90_std": round(
            float(np.std(macro_scores, ddof=1)), 4
        ),
        "mean_error_average": round(
            float(np.mean(mean_errors)), 4
        ),
        "macro_cep90_per_seed": (
            macro_scores.tolist()
        )
    }

    s = results["summary"][method]

    print(
        f"{method:10s} | "
        f"macroCEP90均值={s['macro_cep90_mean']:.4f} | "
        f"标准差={s['macro_cep90_std']:.4f} | "
        f"平均定位误差={s['mean_error_average']:.4f}"
    )


# =====================================
# 5. 配对比较：V5、V6 与自适应几何
# =====================================

print("\n===== 配对差异分析 =====")

paired_results = {}

for candidate in ["v5", "v6"]:

    # 差值 < 0 表示候选方法优于adaptive
    differences = np.array([
        item["metrics"][candidate]["macro_cep90"]
        - item["metrics"]["adaptive"]["macro_cep90"]
        for item in results["seed_results"]
    ])

    n = len(differences)
    mean_diff = float(np.mean(differences))
    std_diff = float(
        np.std(differences, ddof=1)
    )

    # 基于随机种子间差异的探索性t区间
    margin = float(
        t.ppf(0.975, df=n - 1)
        * std_diff / np.sqrt(n)
    )

    lower = mean_diff - margin
    upper = mean_diff + margin

    wins = int(np.sum(differences < 0))
    losses = int(np.sum(differences > 0))
    ties = int(np.sum(differences == 0))

    paired_results[candidate] = {
        "mean_difference": round(mean_diff, 4),
        "confidence_interval_95": [
            round(lower, 4),
            round(upper, 4)
        ],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "differences_per_seed": differences.tolist()
    }

    print(f"\n{candidate} 相对 adaptive：")
    print(
        "平均宏CEP90差值：",
        round(mean_diff, 4)
    )
    print(
        "探索性95%置信区间：",
        (
            round(lower, 4),
            round(upper, 4)
        )
    )
    print(
        "胜/负/平：",
        wins, losses, ties
    )


results["paired_comparisons"] = paired_results


# =====================================
# 6. 识别每个随机种子的最佳方法
# =====================================

print("\n===== 各种子最佳算法 =====")

winner_counts = {
    method: 0
    for method in METHODS
}

for item in results["seed_results"]:

    scores = {
        method:
            item["metrics"][method]["macro_cep90"]
        for method in METHODS
    }

    winner = min(
        scores,
        key=scores.get
    )

    winner_counts[winner] += 1

    print(
        f"seed={item['seed']} | "
        f"best={winner} | "
        f"macroCEP90={scores[winner]:.4f}"
    )

results["winner_counts"] = winner_counts


# =====================================
# 7. 保存CSV与JSON
# =====================================

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "method",
        "macro_cep90_mean",
        "macro_cep90_std",
        "mean_error_average"
    ])

    for method in METHODS:

        item = results["summary"][method]

        writer.writerow([
            method,
            item["macro_cep90_mean"],
            item["macro_cep90_std"],
            item["mean_error_average"]
        ])

save_json()

print("\n===== 最终统计验证完成 =====")
print("JSON：", OUTPUT_JSON)
print("CSV：", OUTPUT_CSV)
print("本程序没有调用DeepSeek API")
