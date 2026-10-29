from transformers import AutoTokenizer


def count_tokens(text: str, model_name: str = "gpt2") -> int:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return len(tokenizer.encode(text))


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = "Retrieval-Augmented Generation (RAG) combines search with LLMs."
    token_ids = tokenizer.encode(text)
    print("Token count:", len(token_ids))
    print("Token IDs:", token_ids)
    print("Decoded:", tokenizer.decode(token_ids))
    print("Tokens:", tokenizer.convert_ids_to_tokens(token_ids))

    batch = tokenizer(
        ["What is RAG?", "Explain vector databases in detail."],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    print("Batch input_ids shape:", tuple(batch["input_ids"].shape))
    print("Attention mask shape:", tuple(batch["attention_mask"].shape))
    print("Long prompt count:", count_tokens("Summarise this: " + "word " * 500))


if __name__ == "__main__":
    main()