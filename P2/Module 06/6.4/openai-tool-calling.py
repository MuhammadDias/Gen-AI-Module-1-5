from openai import OpenAI
import os, json
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
   api_key=os.environ["XKIRO_API_KEY"],
   base_url="https://api.xkiro.com/v1",
)
tools = [
 {
 "type": "function",
 "function": {
 "name": "get_model_info",
    "description": "Returns context window and pricing for a given LLM.",
 "parameters": {
 "type": "object",
 "properties": {
    "model_name": {"type": "string", "description": "Model identifier."}
 },
 "required": ["model_name"]
 }
 }
 }
]
def get_model_info(model_name: str) -> dict:
 db = {
   "qwen/qwen3.7-flash:free": {"context_k": None, "cost_input": None},
 }
 return db.get(model_name, {"error": "unknown model"})
messages = [{"role": "user", "content": "What is qwen/qwen3.7-flash:free's context window?"}]
response = client.chat.completions.create(
 model="qwen/qwen3.7-flash:free",
 tools=tools,
 messages=messages
)
if response.choices[0].finish_reason == "tool_calls":
   tool_call = response.choices[0].message.tool_calls[0]
   args = json.loads(tool_call.function.arguments)
   result = get_model_info(**args)
   messages.append(response.choices[0].message)
   messages.append(
      {
         "role": "tool",
         "tool_call_id": tool_call.id,
         "content": json.dumps(result),
      }
   )
   final = client.chat.completions.create(
      model="qwen/qwen3.7-flash:free",
      messages=messages,
   )
   print(final.choices[0].message.content)
