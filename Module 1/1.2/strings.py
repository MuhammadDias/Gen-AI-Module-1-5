# 1.2 Strings and f-strings for prompt assembly
user_input = "Explain transformers in simple terms"
system_prompt = "You are a helpful AI tutor."

# f-strings: preferred way to assemble prompts
full_prompt = f"System: {system_prompt}\nUser: {user_input}"
print(full_prompt)

# Common string methods
print(user_input.upper())
print(user_input.split())
print(user_input.replace("simple", "plain"))
print("Character count:", len(user_input))

# Multi-line strings for system prompts
prompt = """
You are an expert data scientist.
Answer concisely in bullet points.
"""
print(prompt.strip())
