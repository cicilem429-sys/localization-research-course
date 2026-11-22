
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 读取 .env 文件中的 API Key
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("没有找到 DeepSeek API Key，请检查 .env 文件")

# 2. 创建 DeepSeek 客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# 3. 向 DeepSeek 发送一个数学问题
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "user",
            "content": "请用通俗易懂的语言解释什么是最小二乘法，并给出数学公式。"
        }
    ]
)

# 4. 输出模型的回答
print(response.choices[0].message.content)
