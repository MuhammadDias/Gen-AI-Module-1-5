# 3.3 Parameterised Retry Decorator
import functools
import time
import random

def retry(max_attempts: int = 3, delay: float = 0.1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.05)
def flaky_api_call(prompt: str) -> str:
    if random.random() < 0.6:
        raise ConnectionError("Simulated network error")
    return f"Response to: {prompt}"

if __name__ == "__main__":
    try:
        print(flaky_api_call("Test prompt"))
    except ConnectionError:
        print("Flaky call failed after retries.")
