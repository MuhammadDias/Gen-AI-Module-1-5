from openai import OpenAI
import os
import json
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
	api_key=os.environ["XKIRO_API_KEY"],
	base_url="https://api.xkiro.com/v1",
)
# 1. Define the tool schema
tools = [
 {
 "type": "function",
 "function": {
 "name": "get_model_info",
 "description": "Returns context window size and cost per 1K tokens for a given LLM.",
 "parameters": {
 "type": "object",
 "properties": {
 "model_name": {
 "type": "string",
 "description": "The model identifier, for example 'qwen/qwen3.7-flash:free'."
 }
 },
 "required": ["model_name"]
 }
 }
 }
]
# 2. The actual function the tool will call
def get_model_info(model_name: str) -> dict:
 db = {
 "qwen/qwen3.7-flash:free": {"context_k": None, "cost_input": None, "cost_output": None},
 }
 return db.get(model_name, {"error": f"Unknown model: {model_name}"})
# 3. First API call - model may return a tool_use block
response = client.chat.completions.create(
 model="qwen/qwen3.7-flash:free",
 max_tokens=1024,
 tools=tools,
 messages=[
	 {"role": "user", "content": "What context window and pricing are configured for qwen/qwen3.7-flash:free?"}
 ]
)
# 4. Check if model wants to use a tool
assistant_message = response.choices[0].message
if assistant_message.tool_calls:
 tool_call = assistant_message.tool_calls[0]
 tool_name = tool_call.function.name
 tool_input = json.loads(tool_call.function.arguments)
 # 5. Execute the function
 result = get_model_info(**tool_input)
 print(f"Tool called:{tool_name}({tool_input})")
 print(f"Tool result:{result}")
 # 6. Send tool result back to the model
 final = client.chat.completions.create(
 model="qwen/qwen3.7-flash:free",
 max_tokens=1024,
 tools=tools,
 tool_choice="auto",
 messages=[
 {"role": "user", "content": "What context window and pricing are configured for qwen/qwen3.7-flash:free?"},
 assistant_message,
 {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
 ]
 )
 print("\nFinal answer:")
 print(final.choices[0].message.content)