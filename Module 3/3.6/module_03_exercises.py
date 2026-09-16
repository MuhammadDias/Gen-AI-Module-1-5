# 3.6 Module 03 Exercises
import time
from dataclasses import dataclass
from string import Formatter

class RateLimiter:
    def __init__(self, max_calls_per_minute: int = 5):
        self.interval = 60.0 / max_calls_per_minute
        self.last_call = 0.0

    def check_and_wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()

@dataclass
class PromptTemplate:
    template: str

    def render(self, **kwargs) -> str:
        formatter = Formatter()
        field_names = {fname for _, fname, _, _ in formatter.parse(self.template) if fname is not None}
        missing = field_names - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing placeholders: {missing}")
        return self.template.format(**kwargs)

if __name__ == "__main__":
    pt = PromptTemplate("Hello {name}, role: {role}")
    print(pt.render(name="User", role="Engineer"))
