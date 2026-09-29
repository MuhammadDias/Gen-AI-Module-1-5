# from openai import OpenAI
# import os
# from dotenv import load_dotenv

# load_dotenv()

# client = OpenAI(
#     api_key=os.environ["XKIRO_API_KEY"],
#     base_url="https://api.xkiro.com/v1"
# )

# # Melihat model yang tersedia
# models = client.models.list()

# for model in models.data:
#     print(model.id)

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPEN_ROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

models = client.models.list()

for model in models.data:
    print(model.id)