
import numpy as np

from cluster_consistency import candidate_cluster_spread

anchors = np.array([
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10],
    [5, 15]
], dtype=float)

true_position = np.array([4.0, 6.0])

true_distances = np.linalg.norm(
    anchors - true_position, axis=1
)

noise_levels = [0.1, 0.2, 0.5, 1.0]

def generate_data(rng, trials_per_level):
    features = []
    labels = []

    for sigma in noise_levels:
        for _ in range(trials_per_level):
            distances = true_distances + rng.normal(
                0, sigma, size=len(anchors)
            )
            distances[3] += 8.0

            result = candidate_cluster_spread(
                anchors, distances, cluster_size=4
            )

            features.append(result["cluster_spread"])
            labels.append(sigma)

    return np.array(features), np.array(labels)


# 用不同随机种子生成互不重叠的训练集和测试集
train_rng = np.random.default_rng(42)
test_rng = np.random.default_rng(2026)

train_spreads, train_sigmas = generate_data(
    train_rng, trials_per_level=100
)

test_spreads, test_sigmas = generate_data(
    test_rng, trials_per_level=100
)

# 最小二乘估计比例系数
a = np.dot(train_spreads, train_sigmas) / np.dot(
    train_spreads, train_spreads
)

print("标定系数 a =", round(a, 4))

estimated_sigmas = a * test_spreads

print("\n===== 独立测试集 =====")

for sigma in noise_levels:
    mask = np.isclose(test_sigmas, sigma)
    estimates = estimated_sigmas[mask]

    print(f"\n真实 sigma = {sigma}")
    print(
        "平均估计值 =",
        round(np.mean(estimates), 4)
    )
    print(
        "平均绝对误差 =",
        round(np.mean(np.abs(estimates - sigma)), 4)
    )
    print(
        "估计值标准差 =",
        round(np.std(estimates), 4)
    )

print(
    "\n总体平均绝对误差 =",
    round(
        np.mean(np.abs(estimated_sigmas - test_sigmas)),
        4
    )
)
