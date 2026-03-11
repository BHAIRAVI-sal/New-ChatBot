
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key)

texts = ["hello", "world"]
res = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=texts
)

print(f"Type of res.embeddings: {type(res.embeddings)}")
print(f"Type of res.embeddings[0]: {type(res.embeddings[0])}")
print(f"Type of res.embeddings[0].values: {type(res.embeddings[0].values)}")
print(f"First few values: {res.embeddings[0].values[:5]}")

import json
try:
    json.dumps(res.embeddings[0].values)
    print("JSON serializable: Yes")
except:
    print("JSON serializable: No")

# Check if it's a sequence
from collections.abc import Sequence
print(f"Is sequence: {isinstance(res.embeddings[0].values, Sequence)}")
