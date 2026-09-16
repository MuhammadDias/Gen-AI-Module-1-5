# 3.5 Module and Package Structure Demo
"""
Recommended project layout with pyproject.toml:

my_ai_project/
├── pyproject.toml
├── src/
│   └── my_ai_project/
│       ├── __init__.py
│       ├── config.py
│       ├── clients/
│       │   ├── __init__.py
│       │   ├── anthropic.py
│       │   └── openai.py
│       └── retrieval/
│           ├── __init__.py
│           ├── chunker.py
│           └── embedder.py
└── tests/
"""
def layout_info():
    return "Always use pyproject.toml and src/ layout for modular AI applications."

if __name__ == "__main__":
    print(layout_info())
