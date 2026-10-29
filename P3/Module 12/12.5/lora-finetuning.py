"""Fine-tune a small causal language model with a LoRA adapter."""

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

MODEL_NAME = "distilgpt2"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
    )
    adapted_model = get_peft_model(model, lora_config)
    adapted_model.print_trainable_parameters()

    training_examples = [
        "Question: What is RAG? Answer: RAG retrieves external context for a language model.",
        "Question: What is fine-tuning? Answer: Fine-tuning trains a model on new data.",
        "Question: What is an embedding? Answer: An embedding is a dense vector representing meaning.",
        "Question: What is a vector database? Answer: A vector database stores embeddings for similarity search.",
        "Question: What is LangChain? Answer: LangChain is a framework for composing LLM applications.",
    ]
    dataset = Dataset.from_dict({"text": training_examples})

    def tokenize(batch: dict[str, list[str]]) -> dict[str, object]:
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    output_dir = "lora_output"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = Trainer(
        model=adapted_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    adapter_dir = "lora_adapter"
    adapted_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to {adapter_dir}.")

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tuned_model = PeftModel.from_pretrained(base_model, adapter_dir)
    tuned_model.eval()
    inputs = tokenizer("Question: What is RAG? Answer:", return_tensors="pt")
    with torch.no_grad():
        generated = tuned_model.generate(
            **inputs,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()