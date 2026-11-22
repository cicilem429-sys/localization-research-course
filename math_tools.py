
import numpy as np


def least_squares(x, y):
    """
    给定一组二维数据点，计算最小二乘拟合直线：
        y = a + bx
    """

    # 将输入数据转换为 NumPy 数组
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # 检查数据是否合法
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x 和 y 必须是一维数组")

    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x 和 y 长度必须相同，且至少有两个点")

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("数据不能包含 NaN 或无穷大")

    if np.all(x == x[0]):
        raise ValueError("所有 x 相同，无法唯一确定斜率")

    # 构造矩阵 A，每一行是 [1, x_i]
    A = np.column_stack([np.ones(len(x)), x])

    # 求解最小二乘问题
    coefficients, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    a, b = coefficients

    # 计算预测值和残差平方和
    y_pred = A @ coefficients
    residual_sum = np.sum((y - y_pred) ** 2)

    return {
        "intercept": float(a),
        "slope": float(b),
        "residual_sum": float(residual_sum)
    }


# 下面是测试代码
if __name__ == "__main__":
    result = least_squares(
        x=[1, 2, 3, 4],
        y=[2, 4, 5, 8]
    )

    print("拟合结果：")
    print("截距 a =", result["intercept"])
    print("斜率 b =", result["slope"])
    print("残差平方和 =", result["residual_sum"])


def huber_regression(x, y, delta=1.0, max_iter=1000, tol=1e-8):
    """
    使用 IRLS 算法进行 Huber 鲁棒线性回归。
    拟合模型：y = a + bx
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x 和 y 必须是一维数组")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("数据长度不合法")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("数据必须是有限数值")
    if np.all(x == x[0]):
        raise ValueError("所有 x 相同，无法唯一确定斜率")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta 必须为正的有限数")

    A = np.column_stack([np.ones(len(x)), x])

    # 用普通最小二乘法初始化参数
    beta = np.linalg.lstsq(A, y, rcond=None)[0]

    converged = False
    iterations = 0

    for iteration in range(max_iter):
        residuals = y - A @ beta

        # 计算 Huber 权重
        weights = np.ones_like(residuals)
        mask = np.abs(residuals) > delta
        weights[mask] = delta / np.abs(residuals[mask])

        # 加权最小二乘
        sqrt_w = np.sqrt(weights)
        Aw = A * sqrt_w[:, None]
        yw = y * sqrt_w

        beta_new = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        iterations = iteration + 1

        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            converged = True
            break

        beta = beta_new

    residuals = y - A @ beta
    abs_r = np.abs(residuals)
    huber_loss = np.where(
        abs_r <= delta,
        0.5 * residuals**2,
        delta * (abs_r - 0.5 * delta)
    ).sum()

    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "huber_loss": float(huber_loss),
        "iterations": iterations,
        "converged": converged
    }
