# 2.5 Dict and Set Comprehensions
models = ["gpt-4o", "claude-sonnet-4-5", "gemini-1.5-pro"]
context_windows = [128_000, 200_000, 1_000_000]

model_context = {m: c for m, c in zip(models, context_windows)}
print("Model context:", model_context)

large_context = {m: c for m, c in model_context.items() if c > 150_000}
print("Large context:", large_context)

sentence = "the quick brown fox jumps over the lazy dog"
unique_lengths = {len(w) for w in sentence.split()}
print("Unique word lengths:", sorted(unique_lengths))
