from transformers import pipeline


def main() -> None:
    classifier = pipeline(
        "text-classification",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
    )
    for text, result in zip(
        ["This API is fast and easy to use.", "The latency is terrible."],
        classifier(["This API is fast and easy to use.", "The latency is terrible."]),
    ):
        print(f"[{result['label']} | {result['score']:.3f}] {text}")

    ner = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=-1,
    )
    text = "Anthropic was founded in San Francisco by Dario Amodei."
    for entity in ner(text):
        print(f"{entity['word']!r:25s} -> {entity['entity_group']} ({entity['score']:.3f})")

    generator = pipeline("text-generation", model="distilgpt2", device=-1)
    outputs = generator(
        "The future of AI agents is",
        max_new_tokens=40,
        num_return_sequences=2,
        temperature=0.8,
        do_sample=True,
        pad_token_id=50256,
    )
    for index, output in enumerate(outputs, start=1):
        print(f"Sequence {index}: {output['generated_text']}")

    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    article = (
        "Large language models have changed software development. Developers use AI APIs "
        "to write code, explain concepts, extract information, and reason through problems. "
        "Instruction-tuned models made these capabilities broadly accessible."
    )
    print(summarizer(article, max_length=50, min_length=15, do_sample=False)[0]["summary_text"])


if __name__ == "__main__":
    main()