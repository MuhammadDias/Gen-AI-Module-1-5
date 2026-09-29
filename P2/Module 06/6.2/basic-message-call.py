from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["XKIRO_API_KEY"],
    base_url="https://api.xkiro.com/v1"
)

message = client.chat.completions.create(
    model="qwen/qwen3.7-flash:free",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "What is retrieval-augmented generation?"
        }
    ]
)

# Response text is in the first choice
print(message.choices[0].message.content)

# Usage stats
print(f"Input tokens: {message.usage.prompt_tokens}")
print(f"Output tokens: {message.usage.completion_tokens}")