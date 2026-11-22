
import os
import json

from dotenv import load_dotenv
from openai import OpenAI
from math_tools import least_squares, huber_regression

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("未找到 DeepSeek API Key")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# 固定实验数据，便于复现
X = [1, 2, 3, 4, 5]
Y = [2, 4, 6, 8, 30]

def run_experiment(delta):
    """执行一次 Huber 与普通最小二乘对比实验。"""
    if not isinstance(delta, (int, float)) or not 0.1 <= delta <= 10:
        raise ValueError("delta 必须在 0.1 到 10 之间")

    ols = least_squares(X, Y)
    huber = huber_regression(X, Y, delta=delta)

    return {
        "delta": delta,
        "ols_slope": ols["slope"],
        "huber_slope": huber["slope"],
        "ols_slope_error": abs(ols["slope"] - 2),
        "huber_slope_error": abs(huber["slope"] - 2),
        "huber_intercept_error": abs(huber["intercept"]),
        "converged": huber["converged"]
    }


tools = [{
    "type": "function",
    "function": {
        "name": "run_experiment",
        "description": (
            "在固定的含异常值数据集上，使用指定的 "
            "Huber 阈值 delta 运行对比实验。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delta": {
                    "type": "number",
                    "description": "Huber 损失阈值，范围 0.1 到 10"
                }
            },
            "required": ["delta"]
        }
    }
}]

messages = [
    {
        "role": "system",
        "content": (
            "你是一名数学研究智能体。"
            "你的任务是通过实际调用实验工具，研究 "
            "Huber 阈值对异常值回归效果的影响。"
            "每次只选择一个 delta 进行实验。"
            "根据实验结果选择下一次 delta。"
            "不要编造实验数据。"
            "完成三次实验后给出分析和局限性。"
        )
    },
    {
        "role": "user",
        "content": (
            "研究数据 (1,2),(2,4),(3,6),(4,8),(5,30)。"
            "真实模型为 y=2x。"
            "请进行三次 Huber 阈值实验，"
            "根据上一轮结果选择下一轮参数。"
        )
    }
]

# 最多执行三轮实验
for round_number in range(1, 4):
    print(f"\n===== 第 {round_number} 轮实验 =====")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "run_experiment"}
        }
    )

    message = response.choices[0].message
    messages.append(message)

    if not message.tool_calls:
        raise RuntimeError("模型没有请求实验工具")

    for tool_call in message.tool_calls:
        if tool_call.function.name != "run_experiment":
            raise RuntimeError("模型请求了未知工具")

        args = json.loads(tool_call.function.arguments)
        result = run_experiment(args["delta"])

        print("DeepSeek 选择 delta =", result["delta"])
        print("实验结果：")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, ensure_ascii=False)
        })

# 最终分析
messages.append({
    "role": "user",
    "content": (
        "三次实验已经结束。"
        "请比较实验结果，指出哪个 delta 的斜率误差最小，"
        "并说明实验局限性。"
    )
})

final_response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages
)

print("\n===== 智能体最终研究总结 =====")
print(final_response.choices[0].message.content)
