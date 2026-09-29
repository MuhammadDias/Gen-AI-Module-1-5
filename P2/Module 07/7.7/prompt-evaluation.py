from openai import OpenAI
import json
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.environ["XKIRO_API_KEY"],
    base_url="https://api.xkiro.com/v1",
)


@dataclass
class EvalCase:
    input_text: str
    expected_keywords: list[str]
    must_be_json: bool = False


def evaluate_prompt(system: str, cases: list[EvalCase]) -> dict:
    """Run a prompt against test cases and return pass rate plus details."""
    results = []
    for case in cases:
        response = client.chat.completions.create(
            model="qwen/qwen3.7-flash:free",
            max_tokens=256,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": case.input_text},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        keyword_hit = any(
            keyword.lower() in text.lower()
            for keyword in case.expected_keywords
        )
        json_valid = True
        if case.must_be_json:
            try:
                json.loads(text)
            except json.JSONDecodeError:
                json_valid = False
        passed = keyword_hit and json_valid
        results.append(
            {
                "input": case.input_text[:60],
                "passed": passed,
                "response_preview": text[:80],
            }
        )
    pass_rate = sum(result["passed"] for result in results) / len(results) if results else 0.0
    return {"pass_rate": pass_rate, "results": results}


CLASSIFY_SYSTEM = """Classify the AI task as one of: CLASSIFICATION, GENERATION, RETRIEVAL, EMBEDDING.
Return ONLY the category word."""

test_cases = [
    EvalCase("Predict whether an email is spam.", ["CLASSIFICATION"]),
    EvalCase("Write a product description for headphones.", ["GENERATION"]),
    EvalCase("Find the most relevant documents for a query.", ["RETRIEVAL"]),
    EvalCase("Convert this sentence to a vector.", ["EMBEDDING"]),
    EvalCase("Label customer reviews as positive or negative.", ["CLASSIFICATION"]),
]
report = evaluate_prompt(CLASSIFY_SYSTEM, test_cases)
print(f"Pass rate: {report['pass_rate']:.0%}")
for result in report["results"]:
    status = "PASS" if result["passed"] else "FAIL"
    print(f"  [{status}] {result['input']!r} -> {result['response_preview']!r}")
