# 1.7 Module 01 Exercises
import functools
import time

# Exercise 1: Token cost calculation
def token_cost(tokens: int, model: str) -> float:
    costs = {
        "gpt-4o": 0.0025,
        "claude-sonnet-4-5": 0.003,
        "gemini-1.5-pro": 0.00125
    }
    if model not in costs:
        raise ValueError(f"Unknown model: {model}")
    return (tokens / 1000) * costs[model]

# Exercise 2: Retry decorator
def retry(n: int, sleep_time: float = 0.05):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for _ in range(n):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    time.sleep(sleep_time)
            raise last_err
        return wrapper
    return decorator

# Exercise 3: Temperature label classifier
def temperature_label(t: float) -> str:
    if not (0.0 <= t <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")
    if t <= 0.3:
        return "precise"
    elif t <= 0.7:
        return "balanced"
    else:
        return "creative"

# Exercise 4: Parse pricing string without regex
def parse_pricing_str(data: str) -> tuple[int, float]:
    parts = data.split(",")
    token_str = parts[0].strip().split()[0]
    cost_str = parts[1].strip().split()[0]
    return int(token_str), float(cost_str)

if __name__ == "__main__":
    print("Token Cost (2000 tokens):", token_cost(2000, "gpt-4o"))
    print("Temperature Label (0.5):", temperature_label(0.5))
    tokens, cost = parse_pricing_str("128000 tokens, 0.005 USD per 1K")
    print(f"Parsed -> Tokens: {tokens}, Cost: {cost}")
