import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["XKIRO_API_KEY"],
    base_url="https://api.xkiro.com/v1"
)

stream = client.chat.completions.create(
    model="qwen/qwen3.7-flash:free",
    max_tokens=512,
    messages=[
        {
            "role": "user",
            "content": "List 5 use cases for vector databases."
        }
    ],
    stream=True
)

for chunk in stream:
    text = chunk.choices[0].delta.content

    if text:
        print(text, end="", flush=True)

print()