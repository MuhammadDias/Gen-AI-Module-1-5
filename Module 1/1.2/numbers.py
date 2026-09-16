# 1.2 Numbers and Arithmetic Operations
import math

# Integer arithmetic
tokens_used = 450
tokens_limit = 1024
remaining = tokens_limit - tokens_used
print("Remaining tokens:", remaining)

# Float arithmetic & cost estimation
cost_per_token = 0.000003
total_cost = tokens_used * cost_per_token
print(f"Cost: ${total_cost:.6f}")

# Integer division and modulo
batches = tokens_used // 100
leftover = tokens_used % 100
print(f"Batches: {batches}, Leftover: {leftover}")

# Built-in math utilities
print("Log2(512):", math.log2(512))
print("Ceil(3.1):", math.ceil(3.1))

# Readability underscores
large_number = 1_000_000
print("Large number literal:", large_number)
