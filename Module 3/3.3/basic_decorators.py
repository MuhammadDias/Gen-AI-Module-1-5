# 3.3 Decorators - Logging and Caching
import functools
import time

def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f">>> Calling {func.__name__}")
        res = func(*args, **kwargs)
        print(f"<<< {func.__name__} returned: {res!r}")
        return res
    return wrapper

def cache_result(func):
    _cache: dict = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in _cache:
            _cache[args] = func(*args)
        return _cache[args]
    return wrapper

@log_call
@cache_result
def get_embedding(text: str) -> list[float]:
    time.sleep(0.01)
    return [hash(text) % 100 / 100.0, 0.42, 0.87]

if __name__ == "__main__":
    e1 = get_embedding("RAG")
    e2 = get_embedding("RAG")
    print("Match from cache:", e1 == e2)
