import anthropic, os
from dotenv import load_dotenv
load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
# Poor system prompt - vague, no constraints
WEAK_SYSTEM = "You are an AI assistant."
# Strong system prompt - explicit role, rules, format
STRONG_SYSTEM = """You are a senior Python engineer reviewing code for a production AI pipeline.
Your job:
- Identify bugs, security issues, and performance problems
- Suggest concrete improvements with code examples
- Explain WHY each issue matters
Rules:
- Be direct. Do not pad with compliments.
- If code is correct, say so briefly and move on.
- Always include the corrected code when suggesting a fix.
Format:
Return your review as a numbered list. Each item: Issue → Imp
act → Fix."""
messages = [
 {"role": "user", "content": """Review this function:
def get_user(user_id):
 key = os.getenv('DB_KEY')
 result = requests.get(f'http://db/{user_id}?key={key}')
 return result.json()"""}
]
ponse = client.messages.create(
 model="claude-sonnet-4-5",
 max_tokens=1024,
 system=STRONG_SYSTEM,
 messages=messages,
)
print(response.content[0].text)
