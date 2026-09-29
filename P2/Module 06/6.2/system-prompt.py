import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["XKIRO_API_KEY"],
    base_url="https://api.xkiro.com/v1"
)

message = client.chat.completions.create(
    model="qwen/qwen3.7-flash:free",
    max_tokens=512,
    messages=[
        {
            "role": "system",
            "content": "You are a concise technical writer. Answer in plain English, no jargon."
        },
        {
            "role": "user",
            "content": "Explain what a vector database does."
        }
    ]
)

print(message.choices[0].message.content)