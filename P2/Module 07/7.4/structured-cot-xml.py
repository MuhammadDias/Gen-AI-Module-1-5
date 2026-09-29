from openai import OpenAI
import os
import re
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.environ["XKIRO_API_KEY"],
    base_url="https://api.xkiro.com/v1",
)

SYSTEM = """Solve problems using this exact format:
<thinking>
Step-by-step reasoning here.
</thinking>
<answer>
The final answer only, no reasoning.
</answer>"""

response = client.chat.completions.create(
    model="qwen/qwen3.7-flash:free",
    max_tokens=512,
    messages=[
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "A RAG pipeline retrieves 5 documents, each 400 tokens. "
                "The query is 50 tokens. The model has a 4096 token limit "
                "for context. How many tokens remain for the response?"
            ),
        }
    ],
)

text = response.choices[0].message.content or ""
thinking = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
print("Reasoning:", thinking.group(1).strip() if thinking else "not found")
print("Answer:   ", answer.group(1).strip() if answer else "not found")
