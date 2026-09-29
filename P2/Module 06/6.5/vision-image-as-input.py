from openai import OpenAI
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["XKIRO_API_KEY"],
    base_url="https://api.xkiro.com/v1",
)


# Option A: URL (fastest)
def describe_image_url(url: str) -> str:
    response = client.chat.completions.create(
        model="qwen/qwen3.7-flash:free",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": url}
                },
                {
                    "type": "text",
                    "text": "Describe what you see in this image."
                },
            ],
        }],
    )

    return response.choices[0].message.content or ""


# Option B: base64 (for local files)
def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.standard_b64encode(data).decode()

    ext = Path(path).suffix.lstrip(".").lower()

    media_type = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg"
    }.get(ext, f"image/{ext}")

    data_url = f"data:{media_type};base64,{b64}"

    response = client.chat.completions.create(
        model="qwen/qwen3.7-flash:free",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
            ],
        }],
    )

    return response.choices[0].message.content or ""


# Usage
text = describe_image_url(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Sunrise_over_the_sea.jpg/1280px-Sunrise_over_the_sea.jpg"
)

print(text)