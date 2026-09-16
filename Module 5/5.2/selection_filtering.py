# 5.2 Pandas Selection and Filtering
import pandas as pd

df = pd.DataFrame([
    {"model": "gpt-4o", "context_k": 128, "cost_input": 2.50},
    {"model": "claude-sonnet-4-5", "context_k": 200, "cost_input": 3.00},
    {"model": "gemini-1.5-pro", "context_k": 1000, "cost_input": 1.25},
    {"model": "llama-3.1-70b", "context_k": 128, "cost_input": 0.00},
])

affordable = df[df["cost_input"] < 2.0]
big_and_cheap = df[(df["context_k"] >= 128) & (df["cost_input"] < 2.0)]
print("Affordable:\n", affordable)
print("Big & Cheap:\n", big_and_cheap[["model", "context_k", "cost_input"]])
