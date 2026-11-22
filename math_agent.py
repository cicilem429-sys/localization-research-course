
import os
import json

from dotenv import load_dotenv
from openai import OpenAI
from math_tools import least_squares

# 读取 API Key
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("没有找到 DeepSeek API Key")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# 告诉模型：它可以使用什么工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "least_squares",
            "description": "对二维数据点进行最小二乘直线拟合，返回截距、斜率和残差平方和。",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "数据点的横坐标"
                    },
                    "y": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "数据点的纵坐标"
                    }
                },
                "required": ["x", "y"]
            }
        }
    }
]

messages = [
    {
        "role": "system",
        "content": "你是数学研究助手。需要数值计算时应调用工具，不要编造计算结果。"
    },
    {
        "role": "user",
        "content": (
            "请对数据点 (1,2)、(2,4)、(3,5)、(4,8) "
            "进行最小二乘直线拟合，并解释结果。"
        )
    }
]

# 第一次请求：让 DeepSeek 决定是否调用工具
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

message = response.choices[0].message
messages.append(message)

# 执行模型请求的工具
if message.tool_calls:
    for tool_call in message.tool_calls:
        if tool_call.function.name != "least_squares":
            raise ValueError("模型请求了未知工具")

        arguments = json.loads(tool_call.function.arguments)

        result = least_squares(
            arguments["x"],
            arguments["y"]
        )

        print("Python 数学工具计算结果：")
        print(result)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # 第二次请求：让 DeepSeek 分析工具结果
    final_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )

    print("\nDeepSeek 的数学分析：")
    print(final_response.choices[0].message.content)

else:
    print("模型没有调用工具：")
    print(message.content)
