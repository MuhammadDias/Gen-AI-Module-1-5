# The OpenAI-compatible xKiro API does not expose Anthropic's preflight
# count_tokens operation. Use usage.prompt_tokens after a completion instead.
XKIRO_MODEL = "qwen/qwen3.7-flash:free"
print("Preflight token counting is not available through the xKiro chat API.")
# Cost estimator
PRICING = {
 XKIRO_MODEL: {"input": 0.0, "output": 0.0},
}
def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
 """Return estimated cost in USD."""
 if model not in PRICING:
  raise ValueError(f"Unknown model: {model}")
 p = PRICING[model]
 return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
cost = estimate_cost(XKIRO_MODEL, input_tokens=500, output_tokens=300)
print(f"Estimated cost: ${cost:.6f}")
# Context window limits (always check before sending long documents)
CONTEXT_LIMITS = {
}

def fits_in_context(
    model: str,
    token_count: int,
    context_limit: int,
    reserve_for_output: int = 2048,
) -> bool:
 limit = CONTEXT_LIMITS.get(model, context_limit)
 return token_count + reserve_for_output <= limit