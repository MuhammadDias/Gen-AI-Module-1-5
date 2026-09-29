from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.environ["XKIRO_API_KEY"],
    base_url="https://api.xkiro.com/v1",
)

# Without CoT, the model may jump to an incorrect answer.
DIRECT_PROMPT = (
    "If a model costs $3.00 per million input tokens and $15.00 per million "
    "output tokens, and a request uses 2,400 input tokens and 800 output "
    "tokens, what is the total cost in USD?"
)

# With CoT, the model reasons through each step before answering.
COT_PROMPT = """If a model costs $3.00 per million input tokens and $15.00 per million output tokens,
and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?
Think through this step by step before giving the final answer."""

ZERO_SHOT_COT = """Solve this problem. Think step by step, showing each calculation.
Finally, state: ANSWER: $X.XXXXXX
Problem: A pipeline makes 50 API calls per hour. Each call uses an average of 1,200 input tokens
and 400 output tokens. The model costs $3.00/M input and $15.00/M output.
What is the daily cost?"""

for label, prompt in [
    ("Direct", DIRECT_PROMPT),
    ("CoT", COT_PROMPT),
    ("Zero-shot CoT", ZERO_SHOT_COT),
]:
    response = client.chat.completions.create(
        model="qwen/qwen3.7-flash:free",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    print(f"==={label} ===")
    print((response.choices[0].message.content or "")[:300])
    print()
