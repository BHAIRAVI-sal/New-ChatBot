import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key)

print("Available Models:")
for m in client.models.list():
    print(f"- {m.name}")
