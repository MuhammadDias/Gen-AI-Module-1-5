# 4.3 Environment Variables and Secrets
import os
from dotenv import load_dotenv

load_dotenv()

def get_api_key(provider: str) -> str:
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    env_var = key_map.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")
    key = os.getenv(env_var)
    if not key:
        raise EnvironmentError(f"{env_var} is not set. Add it to your .env file.")
    return key

if __name__ == "__main__":
    print("Environment setup verified.")
