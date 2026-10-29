import os

from dotenv import load_dotenv
from llama_index.core import Document, Settings, SummaryIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

SECTIONS = [
    "Introduction: This report analyses five LLMs across coding, reasoning, and language tasks.",
    "Coding: Claude Sonnet scored highest on the coding benchmark; GPT-4o followed.",
    "Reasoning: GPT-4o led the math reasoning benchmark, closely followed by Claude Sonnet.",
    "Language: Both models performed similarly on language understanding tasks.",
    "Conclusion: Prefer Claude Sonnet for coding and GPT-4o for balanced performance.",
]


def main() -> None:
    api_key = os.environ["OPEN_ROUTER_API_KEY"]
    Settings.llm = OpenAI(
        model=os.getenv("LLM_MODEL", "qwen/qwen3.7-flash:free"),
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )
    Settings.embed_model = OpenAIEmbedding(
        model="openai/text-embedding-3-small",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )
    index = SummaryIndex.from_documents([Document(text=text) for text in SECTIONS])
    response = index.as_query_engine().query("Which model should I use for a coding project?")
    print(response)


if __name__ == "__main__":
    main()