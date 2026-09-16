# 2.5 List Comprehensions
messages = [
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
    {"role": "user", "content": "Give an example."},
]

user_messages = [m["content"] for m in messages if m["role"] == "user"]
print("User messages:", user_messages)

word_counts = [len(m["content"].split()) for m in messages]
print("Word counts:", word_counts)

keywords = [["RAG", "retrieval"], ["LLM", "embedding"], ["vector"]]
flat = [kw for group in keywords for kw in group]
print("Flattened keywords:", flat)
