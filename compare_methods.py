
from math_tools import least_squares, huber_regression

# 正常数据中加入一个明显异常的观测值
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 30]

# 普通最小二乘
ols_result = least_squares(x, y)

# Huber 鲁棒回归
huber_result = huber_regression(x, y, delta=1.0)

print("===== 普通最小二乘法 =====")
print(ols_result)

print("\n===== Huber 鲁棒回归 =====")
print(huber_result)

print("\n===== 与真实模型 y = 2x 比较 =====")
print("真实斜率：2.0")
print("普通最小二乘斜率误差：",
      abs(ols_result["slope"] - 2.0))
print("Huber 回归斜率误差：",
      abs(huber_result["slope"] - 2.0))
