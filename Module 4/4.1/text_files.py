# 4.1 Text Files with pathlib
import pathlib

data_dir = pathlib.Path("data")
data_dir.mkdir(exist_ok=True)

template = """You are a {role}.
Answer the following question concisely.
Question: {question}"""
template_file = data_dir / "qa_prompt.txt"
template_file.write_text(template, encoding="utf-8")

loaded = template_file.read_text(encoding="utf-8")
filled = loaded.format(role="Python tutor", question="What is a generator?")
print(filled)
