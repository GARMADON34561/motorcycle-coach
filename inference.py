# inference.py (place in project root, alongside app.py)

import os
from openai import OpenAI

# Read environment variables with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_inference(prompt: str) -> str:
    """Example inference function that meets the required output format."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Example usage (not executed during validation, but shows structure)
    print(run_inference("Hello from OpenEnv!"))