from openai import OpenAI
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.environ["OPEN_ROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


def embed(
    texts: list[str], model: str = "openai/text-embedding-3-small"
) -> np.ndarray:
    """Embed a list of texts and return an array of shape (n, dim)."""
    response = client.embeddings.create(input=texts, model=model)
    vectors = sorted(response.data, key=lambda item: item.index)
    return np.array([vector.embedding for vector in vectors], dtype=np.float32)


texts = [
    "Retrieval-Augmented Generation combines search with LLMs.",
    "RAG retrieves documents then generates an answer from them.",
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "Fine-tuning trains a model on new data.",
]
embeddings = embed(texts)
print(f"Shape: {embeddings.shape}")
print(f"Norm of first vector: {np.linalg.norm(embeddings[0]):.4f}")
