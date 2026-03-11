import os
import chromadb.utils.embedding_functions as embedding_functions
from google import genai
from dotenv import load_dotenv

load_dotenv()

class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, client, model_name="models/gemini-embedding-001"):
        self.client = client
        self.model_name = model_name

    def __call__(self, input):
        # input is a list of strings
        if isinstance(input, str):
            input = [input]
        res = self.client.models.embed_content(
            model=self.model_name,
            contents=input
        )
        return [[float(v) for v in e.values] for e in res.embeddings]

    @property
    def name(self):
        return self.model_name

gemini_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key)
ef = GeminiEmbeddingFunction(client)

import chromadb
db = chromadb.EphemeralClient()
col = db.create_collection("test", embedding_function=ef)
print("Collection created.")

print("Adding documents...")
col.add(ids=["1"], documents=["hello"])
print("Documents added.")

print("Querying...")
try:
    res = col.query(query_texts=["hello"], n_results=1)
    print("Success!")
    print(res)
except Exception as e:
    print(f"FAILED with error: {e}")
    import traceback
    traceback.print_exc()
