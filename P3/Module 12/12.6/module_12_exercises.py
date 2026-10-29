from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Exercise 1: compare latency and generated character count across two pipelines.
@dataclass(frozen=True)
class PipelineMetrics:
    model: str
    prompt_count: int
    average_latency_ms: float
    average_output_chars: float


def benchmark_pipelines(
    pipelines: dict[str, Callable[[str], Any]],
    prompts: Sequence[str],
    clock: Callable[[], float] = time.perf_counter,
) -> list[PipelineMetrics]:
    if not prompts:
        raise ValueError("At least one prompt is required")
    reports = []
    for model_name, generate in pipelines.items():
        latencies = []
        output_lengths = []
        for prompt in prompts:
            started = clock()
            output = generate(prompt)
            latencies.append((clock() - started) * 1000)
            if isinstance(output, list) and output and isinstance(output[0], dict):
                text = str(output[0].get("generated_text", ""))
            else:
                text = str(output)
            output_lengths.append(len(text))
        reports.append(
            PipelineMetrics(
                model=model_name,
                prompt_count=len(prompts),
                average_latency_ms=sum(latencies) / len(latencies),
                average_output_chars=sum(output_lengths) / len(output_lengths),
            )
        )
    return reports


# Exercise 2: keep texts whose tokenizer count is within the configured budget.
def filter_by_token_budget(
    texts: Sequence[str],
    model_name: str,
    token_limit: int,
    count_tokens: Callable[[str, str], int] | None = None,
) -> list[tuple[str, int]]:
    if token_limit < 0:
        raise ValueError("token_limit must be zero or greater")
    if count_tokens is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        count_tokens = lambda text, _model: len(tokenizer.encode(text))
    accepted = []
    for text in texts:
        count = count_tokens(text, model_name)
        if count < 0:
            raise ValueError("Token counter returned a negative count")
        if count <= token_limit:
            accepted.append((text, count))
    return accepted


# Exercise 3: extract unique NER values by entity category.
def extract_entities_from_corpus(
    texts: Sequence[str],
    ner_pipeline: Callable[[str], Sequence[dict[str, Any]]],
) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for text in texts:
        for entity in ner_pipeline(text):
            category = entity.get("entity_group") or entity.get("entity")
            category = {
                "PER": "PERSON",
                "GPE": "LOC",
                "LOCATION": "LOC",
                "ORGANIZATION": "ORG",
            }.get(category, category)
            value = str(entity.get("word", "")).strip()
            if category in {"PERSON", "ORG", "LOC"} and value:
                grouped[category].add(value)
    return {category: sorted(grouped.get(category, set())) for category in ("PERSON", "ORG", "LOC")}


TRAINING_EXAMPLES = [
    ("What does RAG stand for?", "Retrieval-Augmented Generation."),
    ("What does retrieval do in RAG?", "It finds relevant external documents for a query."),
    ("What is the generation step?", "The language model writes an answer using retrieved context."),
    ("What is an embedding?", "A numerical vector representing data such as text."),
    ("Why use embeddings?", "They make semantic similarity search possible."),
    ("What is a vector database?", "A database that stores vectors and supports similarity search."),
    ("What is a retriever?", "A component that finds relevant documents for a query."),
    ("What is a document chunk?", "A smaller section of a larger source document."),
    ("Why split documents?", "Smaller chunks fit model context and improve retrieval granularity."),
    ("What is chunk overlap?", "Shared text between adjacent chunks that preserves boundary context."),
    ("What is cosine similarity?", "A measure comparing the direction of two vectors."),
    ("What is top-k retrieval?", "Returning the k highest-ranked matching results."),
    ("What is metadata filtering?", "Restricting retrieval using document attributes."),
    ("Does RAG update model weights?", "No. RAG retrieves external information at inference time."),
    ("How does fine-tuning differ from RAG?", "Fine-tuning updates model weights; RAG retrieves external context."),
    ("What is grounding?", "Supporting an answer with evidence from supplied sources."),
    ("What is a hallucination?", "An unsupported or fabricated model output."),
    ("How can RAG reduce hallucinations?", "Provide relevant evidence and instruct the model to use it."),
    ("What does an index do?", "It organizes data to support efficient retrieval."),
    ("What should an answer do when context is missing?", "It should state that the available context is insufficient."),
]


def verify_finetuned_answers(
    generate: Callable[[str], str],
    examples: Sequence[tuple[str, str]] = TRAINING_EXAMPLES,
    minimum_keyword_coverage: float = 0.3,
) -> list[dict[str, Any]]:
    if not 0 <= minimum_keyword_coverage <= 1:
        raise ValueError("minimum_keyword_coverage must be between 0 and 1")
    results = []
    for question, expected in examples:
        actual = generate(f"Question: {question}\nAnswer:")
        expected_terms = {word.lower().strip(".,?!:") for word in expected.split() if len(word) > 3}
        actual_terms = {word.lower().strip(".,?!:") for word in actual.split()}
        coverage = len(expected_terms & actual_terms) / len(expected_terms) if expected_terms else 1.0
        results.append(
            {
                "question": question,
                "expected": expected,
                "actual": actual,
                "keyword_coverage": coverage,
                "passed": coverage >= minimum_keyword_coverage,
            }
        )
    return results


# Exercise 4: prepare 20 Q&A samples, train a LoRA adapter, save, and test generation.
def train_lora_adapter(
    output_dir: str | Path = "lora_rag_adapter",
    epochs: int = 3,
    model_name: str = "distilgpt2",
) -> list[dict[str, Any]]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    adapter = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            target_modules=["c_attn"],
            lora_dropout=0.1,
            bias="none",
        ),
    )
    adapter.print_trainable_parameters()
    formatted = [f"Question: {question}\nAnswer: {answer}{tokenizer.eos_token}" for question, answer in TRAINING_EXAMPLES]
    dataset = Dataset.from_dict({"text": formatted})

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    training_args = TrainingArguments(
        output_dir=str(Path(output_dir) / "training"),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = Trainer(
        model=adapter,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    output_path = Path(output_dir)
    adapter.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    base = AutoModelForCausalLM.from_pretrained(model_name)
    tuned = PeftModel.from_pretrained(base, output_path)
    tuned.eval()

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = tuned.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

    return verify_finetuned_answers(generate)


def run_local_tests() -> None:
    reports = benchmark_pipelines(
        {"first": lambda prompt: prompt + " answer", "second": lambda prompt: prompt},
        ["one", "two"],
        clock=iter(range(20)).__next__,
    )
    assert len(reports) == 2 and reports[0].prompt_count == 2

    filtered = filter_by_token_budget(
        ["a", "bb", "ccc"],
        "mock",
        2,
        count_tokens=lambda text, _model: len(text),
    )
    assert filtered == [("a", 1), ("bb", 2)]

    entities = extract_entities_from_corpus(
        ["text1", "text2"],
        lambda text: [
            {"entity_group": "PER", "word": "Ada"},
            {"entity_group": "ORG", "word": "Lab"},
            {"entity_group": "MISC", "word": "ignored"},
        ] if text == "text1" else [{"entity_group": "PERSON", "word": "Ada"}],
    )
    assert entities == {"PERSON": ["Ada"], "ORG": ["Lab"], "LOC": []}

    checks = verify_finetuned_answers(
        lambda _prompt: "Retrieval-Augmented Generation retrieves relevant external documents for a query.",
        examples=[TRAINING_EXAMPLES[0]],
    )
    assert checks[0]["passed"] is True


if __name__ == "__main__":
    run_local_tests()
    print("Module 12 exercise tests passed (mocked; no model downloads).")