
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from localization_experiment import run_localization_experiment

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("请先在 .env 中配置 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

tool_definition = {
    "type": "function",
    "function": {
        "name": "run_localization_experiment",
        "description": (
            "运行二维 Huber 鲁棒定位实验，"
            "返回留一交叉验证分数及定位估计。"
            "不提供真实目标位置或真实定位误差。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delta": {
                    "type": "number",
                    "description": "Huber 阈值，范围 0.1 到 10"
                }
            },
            "required": ["delta"]
        }
    }
}

messages = [
    {
        "role": "system",
        "content": (
            "你是一名研究鲁棒非线性定位的数学研究智能体。"
            "你的任务是通过实验探索 Huber 阈值对定位的影响。"
            "你不知道真实目标位置，不能假设验证分数"
            "一定等于真实定位误差。"
            "每一轮根据之前的实验结果选择新的 delta。"
            "不得重复已经测试的参数。"
            "最后客观总结发现、局限性和下一步建议。"
        )
    },
    {
        "role": "user",
        "content": (
            "请开展三轮二维鲁棒定位实验。"
            "自主选择三个不同的 Huber 阈值，"
            "尽量探索不同参数范围。"
            "目标是降低留一验证分数，"
            "但不要声称它必然代表真实定位精度。"
        )
    }
]

history = []

for round_index in range(3):
    print(f"\n========== 第 {round_index + 1} 轮 ==========")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=[tool_definition],
        tool_choice={
            "type": "function",
            "function": {
                "name": "run_localization_experiment"
            }
        }
    )

    assistant_message = response.choices[0].message
    messages.append(assistant_message.model_dump(exclude_none=True))

    if not assistant_message.tool_calls:
        raise RuntimeError("模型没有调用实验工具")

 
    # 每轮只接受一次有效实验
    accepted = False

    for call in assistant_message.tool_calls:
        try:
            args = json.loads(call.function.arguments)
            delta = float(args["delta"])

            if accepted:
                result = {
                    "error": "本轮实验名额已用完"
                }

            elif any(
                abs(delta - h["delta"]) < 1e-9
                for h in history
            ):
                result = {
                    "error": "该参数已经测试过"
                }

            else:
                result = run_localization_experiment(delta)
                history.append(result)
                accepted = True

                print("DeepSeek 选择的 delta：", delta)
                print("实验结果：", result)

        except (ValueError, RuntimeError, KeyError) as exc:
            result = {"error": str(exc)}

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(
                result, ensure_ascii=False
            )
        })


# 最终研究总结
messages.append({
    "role": "user",
    "content": (
        "实验阶段结束。请根据已经执行的实验，"
        "比较验证分数，指出当前最优候选参数，"
        "并说明实验局限性。不要再调用工具。"
    )
})

final_response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages
)

print("\n========== 智能体研究总结 ==========")
print(final_response.choices[0].message.content)

# 保存实验记录
with open("localization_history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

print("\n实验记录已保存至 localization_history.json")
