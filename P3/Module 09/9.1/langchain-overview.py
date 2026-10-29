"""LangChain composes prompts, models, parsers, and retrieval into runnable apps."""

CORE_BUILDING_BLOCKS = {
    "prompt": "Shapes model input",
    "model": "Produces a response",
    "parser": "Converts a response to the required output type",
    "retriever": "Finds relevant documents for a query",
}


if __name__ == "__main__":
    for component, purpose in CORE_BUILDING_BLOCKS.items():
        print(f"{component}: {purpose}")