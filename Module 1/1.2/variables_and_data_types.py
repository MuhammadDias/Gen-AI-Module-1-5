# 1.2 Variables and Data Types
# Core scalar types with PEP 484 type hints
model_name: str = "claude-sonnet-4-5"
temperature: float = 0.7
max_tokens: int = 1024
is_streaming: bool = True

print("model_name type:", type(model_name))
print("temperature type:", type(temperature))

# None represents the absence of a value
response = None
print("is None check:", response is None)
