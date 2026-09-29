import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["XKIRO_API_KEY"],
    base_url="https://api.xkiro.com/v1"
)


def chat(system: str) -> None:
    """Simple interactive multi-turn chat loop."""
    history = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            break

        history.append({
            "role": "user",
            "content": user_input
        })

        response = client.chat.completions.create(
            model="qwen/qwen3.7-flash:free",
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                *history
            ]
        )

        assistant_text = response.choices[0].message.content

        history.append({
            "role": "assistant",
            "content": assistant_text
        })

        print(f"Qwen: {assistant_text}\n")


chat(system="You are a helpful Python tutor.")