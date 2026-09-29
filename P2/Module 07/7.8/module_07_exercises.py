from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "qwen/qwen3.7-flash:free"
XKIRO_BASE_URL = "https://api.xkiro.com/v1"


def create_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["XKIRO_API_KEY"],
        base_url=XKIRO_BASE_URL,
    )


# Exercise 1: compare three code-review prompt levels on the same five snippets.
REVIEW_PROMPTS = {
    "basic": "You are a code review assistant. Identify bugs in the supplied code and suggest a fix.",
    "intermediate": """You are a Python code reviewer. Identify correctness, security, and reliability issues.
For every finding, cite the relevant code, explain its impact, and give a concrete fix.
Do not report stylistic preferences as bugs.""",
    "expert": """You are a senior Python engineer reviewing production code. Inspect control flow,
input validation, resource lifecycle, security boundaries, failure modes, and testability.
Report only actionable findings. For each finding give severity, exact code evidence,
impact, a minimal corrected example, and a focused test that would catch it. Avoid
inventing issues and state when the snippet is correct.""",
}


@dataclass(frozen=True)
class ReviewCase:
    name: str
    code: str
    expected_concepts: tuple[str, ...]


CODE_REVIEW_CASES = (
    ReviewCase(
        "http_status_and_timeout",
        """import requests
def fetch(url):
    response = requests.get(url)
    return response.json()""",
        ("timeout", "status", "raise_for_status"),
    ),
    ReviewCase(
        "mutable_default",
        """def add_tag(tag, tags=[]):
    tags.append(tag)
    return tags""",
        ("mutable default", "default argument", "none"),
    ),
    ReviewCase(
        "resource_lifecycle",
        """def read_config(path):
    handle = open(path)
    return handle.read()""",
        ("with", "context manager", "close"),
    ),
    ReviewCase(
        "sql_injection",
        """def find_user(cursor, username):
    query = f\"SELECT * FROM users WHERE name = '{username}'\"
    cursor.execute(query)
    return cursor.fetchone()""",
        ("parameterized", "parameter", "injection"),
    ),
    ReviewCase(
        "swallowed_exception",
        """def load_data(path):
    try:
        return parse_file(path)
    except Exception:
        return {}""",
        ("exception", "silent", "log", "narrow"),
    ),
)


def score_review(response: str, expected_concepts: tuple[str, ...]) -> tuple[int, int]:
    """Return heuristic concept hits and total concepts (not an LLM-grade score)."""
    normalized = response.lower()
    hits = sum(concept in normalized for concept in expected_concepts)
    return hits, len(expected_concepts)


def evaluate_review_prompts(
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each prompt on the identical five-case set.

    The quality score is a simple keyword-coverage heuristic for comparing runs;
    it is not a substitute for a human review of the findings.
    """
    api_client = client or create_client()
    detail_rows = []
    for prompt_level, system_prompt in REVIEW_PROMPTS.items():
        for case in CODE_REVIEW_CASES:
            response = api_client.chat.completions.create(
                model=model,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Review this code:\n```python\n{case.code}\n```",
                    },
                ],
            )
            response_text = response.choices[0].message.content or ""
            hits, total = score_review(response_text, case.expected_concepts)
            detail_rows.append(
                {
                    "prompt_level": prompt_level,
                    "case": case.name,
                    "quality_score": hits / total if total else 1.0,
                    "matched_concepts": hits,
                    "expected_concepts": total,
                    "response_text": response_text,
                }
            )

    details = pd.DataFrame(detail_rows)
    summary = (
        details.groupby("prompt_level", as_index=False)
        .agg(
            mean_quality_score=("quality_score", "mean"),
            cases_passed=("quality_score", lambda scores: int((scores == 1).sum())),
            cases_evaluated=("case", "count"),
        )
        .sort_values("mean_quality_score", ascending=False)
        .reset_index(drop=True)
    )
    return details, summary


# Exercise 2: a JSON-persisted library of versioned prompt templates.
@dataclass
class PromptTemplate:
    name: str
    system: str
    user: str
    version: str = "1.0"
    required_vars: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        combined = self.system + self.user
        self.required_vars = list(
            dict.fromkeys(
                field_name
                for _, field_name, _, _ in Formatter().parse(combined)
                if field_name is not None
            )
        )

    def render(self, **values: Any) -> tuple[str, str]:
        missing = set(self.required_vars) - values.keys()
        if missing:
            raise ValueError(f"Missing template variables: {sorted(missing)}")
        return self.system.format(**values), self.user.format(**values)


class PromptLibrary:
    def __init__(self) -> None:
        self.templates: dict[str, PromptTemplate] = {}
        self.last_evaluated_versions: dict[str, str] = {}

    def add(self, template: PromptTemplate) -> None:
        self.templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        try:
            return self.templates[name]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt template: {name}") from exc

    def record_evaluation(self, name: str, version: str | None = None) -> None:
        template = self.get(name)
        self.last_evaluated_versions[name] = version or template.version

    def save(self, path: str | Path) -> None:
        data = {
            "templates": {
                name: asdict(template) for name, template in self.templates.items()
            },
            "last_evaluated_versions": self.last_evaluated_versions,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PromptLibrary:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        library = cls()
        library.templates = {
            name: PromptTemplate(**template_data)
            for name, template_data in data.get("templates", {}).items()
        }
        library.last_evaluated_versions = data.get("last_evaluated_versions", {})
        return library


# Exercise 3: parse JSON, remove markdown fences, then use xKiro for repair.
def _parse_json_object(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("The parsed JSON value must be an object")
    return value


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def safe_json_parse(text: str, client: OpenAI | None = None) -> dict[str, Any]:
    """Parse JSON, retry after fence removal, then request a JSON-only repair."""
    try:
        return _parse_json_object(text)
    except (json.JSONDecodeError, ValueError):
        pass

    cleaned = _strip_markdown_fences(text)
    try:
        return _parse_json_object(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    api_client = client or create_client()
    response = api_client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_tokens=512,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Repair the input into one valid JSON object. Return only raw JSON, with no markdown or explanation.",
            },
            {"role": "user", "content": cleaned},
        ],
    )
    repaired = response.choices[0].message.content or ""
    repaired = _strip_markdown_fences(repaired)
    return _parse_json_object(repaired)


# Exercise 4: ask for step-by-step ranking and validate against computed means.
SCORE_TEST_CASES: tuple[dict[str, dict[str, float]], ...] = (
    {
        "Model-A": {"math": 95, "coding": 90, "reasoning": 92},
        "Model-B": {"math": 82, "coding": 96, "reasoning": 84},
        "Model-C": {"math": 88, "coding": 85, "reasoning": 90},
        "Model-D": {"math": 78, "coding": 82, "reasoning": 80},
        "Model-E": {"math": 91, "coding": 86, "reasoning": 88},
        "Model-F": {"math": 84, "coding": 89, "reasoning": 87},
        "Model-G": {"math": 75, "coding": 93, "reasoning": 79},
        "Model-H": {"math": 87, "coding": 80, "reasoning": 85},
        "Model-I": {"math": 80, "coding": 77, "reasoning": 83},
        "Model-J": {"math": 92, "coding": 84, "reasoning": 89},
    },
    {
        "Model-A": {"math": 74, "coding": 88, "reasoning": 81},
        "Model-B": {"math": 98, "coding": 91, "reasoning": 95},
        "Model-C": {"math": 86, "coding": 84, "reasoning": 89},
        "Model-D": {"math": 83, "coding": 79, "reasoning": 82},
        "Model-E": {"math": 89, "coding": 93, "reasoning": 90},
        "Model-F": {"math": 81, "coding": 87, "reasoning": 85},
        "Model-G": {"math": 92, "coding": 76, "reasoning": 88},
        "Model-H": {"math": 78, "coding": 82, "reasoning": 80},
        "Model-I": {"math": 85, "coding": 90, "reasoning": 84},
        "Model-J": {"math": 80, "coding": 86, "reasoning": 79},
    },
    {
        "Model-A": {"math": 88, "coding": 79, "reasoning": 85},
        "Model-B": {"math": 86, "coding": 83, "reasoning": 88},
        "Model-C": {"math": 97, "coding": 95, "reasoning": 96},
        "Model-D": {"math": 82, "coding": 87, "reasoning": 84},
        "Model-E": {"math": 79, "coding": 91, "reasoning": 82},
        "Model-F": {"math": 90, "coding": 85, "reasoning": 89},
        "Model-G": {"math": 84, "coding": 78, "reasoning": 81},
        "Model-H": {"math": 76, "coding": 88, "reasoning": 83},
        "Model-I": {"math": 81, "coding": 80, "reasoning": 77},
        "Model-J": {"math": 87, "coding": 92, "reasoning": 86},
    },
)


def expected_model_ranking(
    scores: dict[str, dict[str, float]],
) -> list[dict[str, float | str]]:
    averages = {
        model: sum(task_scores.values()) / len(task_scores)
        for model, task_scores in scores.items()
        if task_scores
    }
    return [
        {"model": model, "average_score": round(average, 2)}
        for model, average in sorted(
            averages.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def build_cot_prompt(scores: dict[str, dict[str, float]]) -> str:
    return f"""Rank these models using the mean of their scores across all three tasks.
Think through the calculations before deciding the order. Give a recommendation
in exactly two sentences. Return only a JSON object with this shape:
{{"ranking": [{{"model": "name", "average_score": 0.0}}], "recommendation": "Two sentences."}}
Scores (each model has three task scores):
{json.dumps(scores, indent=2)}"""


def rank_models_with_cot(
    scores: dict[str, dict[str, float]],
    client: OpenAI | None = None,
) -> dict[str, Any]:
    api_client = client or create_client()
    response = api_client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_tokens=900,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Follow the requested scoring rule exactly and return valid JSON.",
            },
            {"role": "user", "content": build_cot_prompt(scores)},
        ],
    )
    raw = _strip_markdown_fences(response.choices[0].message.content or "")
    result = _parse_json_object(raw)
    validate_cot_result(scores, result)
    return result


def validate_cot_result(
    scores: dict[str, dict[str, float]],
    result: dict[str, Any],
) -> None:
    expected = expected_model_ranking(scores)
    ranking = result.get("ranking")
    recommendation = result.get("recommendation")
    if not isinstance(ranking, list) or len(ranking) != len(expected):
        raise ValueError("Ranking must contain every model exactly once")
    if [item.get("model") for item in ranking] != [item["model"] for item in expected]:
        raise ValueError("Model order does not match the descending mean scores")
    for actual, target in zip(ranking, expected):
        if abs(float(actual["average_score"]) - float(target["average_score"])) > 0.02:
            raise ValueError(f"Incorrect average score for {target['model']}")
    if not isinstance(recommendation, str) or len(
        [sentence for sentence in re.split(r"(?<=[.!?])\s+", recommendation.strip()) if sentence]
    ) != 2:
        raise ValueError("Recommendation must contain exactly two sentences")


def verify_cot_prompt_on_three_inputs(
    client: OpenAI | None = None,
) -> list[dict[str, Any]]:
    """Run the CoT prompt on all three score sets and validate each result."""
    return [rank_models_with_cot(scores, client=client) for scores in SCORE_TEST_CASES]


def run_local_tests() -> None:
    library = PromptLibrary()
    template = PromptTemplate(
        name="qa",
        system="You are a {domain} expert.",
        user="Answer: {question}",
        version="1.1",
    )
    library.add(template)
    library.record_evaluation("qa")
    assert template.render(domain="Python", question="What is a list?")[0] == "You are a Python expert."
    temporary_path = Path("prompt-library-test.json")
    try:
        library.save(temporary_path)
        loaded = PromptLibrary.load(temporary_path)
        assert loaded.last_evaluated_versions == {"qa": "1.1"}
        assert loaded.get("qa").version == "1.1"
    finally:
        temporary_path.unlink(missing_ok=True)

    assert safe_json_parse('{"valid": true}') == {"valid": True}
    assert safe_json_parse('```json\n{"valid": true}\n```') == {"valid": True}
    for scores in SCORE_TEST_CASES:
        expected = expected_model_ranking(scores)
        validate_cot_result(
            scores,
            {
                "ranking": expected,
                "recommendation": "Choose the highest average score. Validate this choice against task-specific requirements.",
            },
        )


def run_api_exercises() -> None:
    client = create_client()
    _, summary = evaluate_review_prompts(client=client)
    print("Code review prompt comparison:")
    print(summary.to_string(index=False))
    print("\nJSON repair:")
    print(safe_json_parse('{"model": "Qwen", score: 92}', client=client))
    print("\nCoT ranking checks:")
    for case_number, result in enumerate(
        verify_cot_prompt_on_three_inputs(client=client), start=1
    ):
        print(f"Input {case_number}: top model = {result['ranking'][0]['model']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-api-exercises",
        action="store_true",
        help="Run the prompt comparisons and model calls using xKiro (uses API quota).",
    )
    arguments = parser.parse_args()
    run_local_tests()
    print("Local Module 07 exercise tests passed.")
    if arguments.run_api_exercises:
        run_api_exercises()
    else:
        print("Use --run-api-exercises to run the xKiro-backed tasks.")
