
import os
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from dotenv import load_dotenv

load_dotenv()

class GeminiEmbeddingFunction:
    def __init__(self, client, model_name="models/gemini-embedding-001"):
        self.client = client
        self.model_name = model_name
        # self.name = model_name # This caused "'str' object is not callable"

    def __call__(self, texts):
        return [[0.1] * 768 for _ in texts]

gemini_key = os.getenv("GEMINI_API_KEY")
client_genai = genai.Client(api_key=gemini_key)
emb_fn = GeminiEmbeddingFunction(client_genai)

client_chroma = chromadb.PersistentClient(path="./test_chroma")
try:
    print("Run 1: Creating collection...")
    col = client_chroma.get_or_create_collection("test_col", embedding_function=emb_fn)
    print("Success 1!")
    
    # Simulate a second run without deleting the folder
    print("Run 2: Getting existing collection...")
    col2 = client_chroma.get_or_create_collection("test_col", embedding_function=emb_fn)
    print("Success 2!")
except Exception as e:
    print(f"Failed with: {e}")
    import traceback
    traceback.print_exc()
finally:
    import shutil
    if os.path.exists("./test_chroma"):
        shutil.rmtree("./test_chroma")
