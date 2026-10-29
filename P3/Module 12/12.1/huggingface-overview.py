"""Transformers provides pipelines, model/tokenizer loaders, and a Trainer API."""

USE_CASES = {
    "privacy or offline inference": "Run an open model locally",
    "high-volume inference": "Compare hardware cost with hosted per-token pricing",
    "domain adaptation": "Fine-tune with PEFT/LoRA where appropriate",
    "fast prototyping": "Start with the pipeline API",
}


if __name__ == "__main__":
    for need, approach in USE_CASES.items():
        print(f"{need}: {approach}")