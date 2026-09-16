# 2.3 Sets - Deduplication and Set Operations
retrieved_doc_ids = ["doc_3", "doc_1", "doc_3", "doc_7", "doc_1"]
unique_ids = set(retrieved_doc_ids)
print("Unique IDs:", unique_ids)

gpt4_topics = {"coding", "math", "reasoning", "vision"}
claude_topics = {"coding", "writing", "reasoning", "safety"}

print("Both (Intersection):", gpt4_topics & claude_topics)
print("Either (Union):", gpt4_topics | claude_topics)
print("GPT only (Difference):", gpt4_topics - claude_topics)
