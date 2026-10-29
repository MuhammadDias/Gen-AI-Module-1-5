import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class ModelComparison(BaseModel):
    models: list[str] = Field(description="Model names compared")
    best_for_cost: str = Field(description="Most cost-effective model")
    best_for_quality: str = Field(description="Highest quality model")
    recommendation: str = Field(description="Overall recommendation")


def main() -> None:
    model = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "qwen/qwen3.7-flash:free"),
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    structured_model = model.with_structured_output(ModelComparison)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI model comparison expert."),
            ("human", "Compare these models for a startup on a budget: {models}"),
        ]
    )
    result = (prompt | structured_model).invoke(
        {"models": "gpt-4o, gpt-4o-mini, claude-sonnet-4-5"}
    )
    print(f"Best for cost: {result.best_for_cost}")
    print(f"Best for quality: {result.best_for_quality}")
    print(f"Recommendation: {result.recommendation}")


if __name__ == "__main__":
    main()