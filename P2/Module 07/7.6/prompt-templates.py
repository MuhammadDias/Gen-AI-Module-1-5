from dataclasses import dataclass, field
from string import Formatter
from typing import Any


@dataclass
class PromptTemplate:
    """A reusable, versioned prompt template."""

    name: str
    system: str
    user: str
    version: str = "1.0"
    required_vars: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        formatter = Formatter()
        combined = self.system + self.user
        self.required_vars = [
            field_name
            for _, field_name, _, _ in formatter.parse(combined)
            if field_name is not None
        ]

    def render(self, **kwargs: Any) -> tuple[str, str]:
        """Return rendered system and user prompts; fail on missing variables."""
        missing = set(self.required_vars) - set(kwargs)
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self.system.format(**kwargs), self.user.format(**kwargs)


QA_TEMPLATE = PromptTemplate(
    name="question_answering",
    version="1.2",
    system="""You are a {domain} expert. Answer questions accurately and concisely.
Cite sources when possible. If you are unsure, say so.""",
    user="Question: {question}\n\nContext:\n{context}",
)

SUMMARY_TEMPLATE = PromptTemplate(
    name="document_summary",
    version="1.0",
    system="You are a technical writer. Summarise documents clearly for a {audience} audience.",
    user="Summarise the following in {max_sentences} sentences or fewer:\n\n{document}",
)

system, user = QA_TEMPLATE.render(
    domain="machine learning",
    question="What is the vanishing gradient problem?",
    context="Gradients in deep networks are computed via backpropagation...",
)
print("System:", system)
print("User:", user)
print("Required vars:", QA_TEMPLATE.required_vars)
