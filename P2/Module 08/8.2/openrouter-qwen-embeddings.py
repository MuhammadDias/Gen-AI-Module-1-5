import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.environ["OPEN_ROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

result = client.embeddings.create(
    input=["What is RAG?", "Explain vector databases."],
    model="nvidia/nemotron-3-embed-1b:free"
)
vectors = sorted(result.data, key=lambda item: item.index)
embeddings = np.array([item.embedding for item in vectors], dtype=np.float32)
print(f"Shape: {embeddings.shape}")
print(f"Token usage: {result.usage.prompt_tokens}")