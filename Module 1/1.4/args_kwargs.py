# 1.4 Functions - *args and **kwargs
def log_messages(*messages: str) -> None:
    for msg in messages:
        print(f"[LOG] {msg}")

log_messages("Starting", "Loading model", "Done")

def create_api_payload(model: str, **kwargs) -> dict:
    payload = {"model": model}
    payload.update(kwargs)
    return payload

payload = create_api_payload(
    "claude-sonnet-4-5",
    max_tokens=1024,
    temperature=0.3,
    stream=True,
)
print("API Payload:", payload)
