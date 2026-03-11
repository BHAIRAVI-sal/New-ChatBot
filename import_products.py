import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, client, model_name="models/gemini-embedding-001"):
        self.client = client
        self.model_name = model_name

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        res = self.client.models.embed_content(
            model=self.model_name,
            contents=input
        )
        return [[float(v) for v in e.values] for e in res.embeddings]

    def name(self):
        return self.model_name

def import_products():
    # 1. Load the CSV
    print("Loading products.csv...")
    df = pd.read_csv('products.csv')
    
    # Fill NaN values to avoid errors
    df = df.fillna('')
    
    # 2. Initialize ChromaDB
    # We will use a persistent client so the data stays on disk
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 3. Setup Embedding Function
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print("Using Gemini embeddings (models/gemini-embedding-001)...")
        client_genai = genai.Client(api_key=gemini_key)
        emb_fn = GeminiEmbeddingFunction(client_genai)
    else:
        print("GEMINI_API_KEY not found. Using default Chroma embeddings...")
        emb_fn = embedding_functions.DefaultEmbeddingFunction()

    # 4. Create or Get Collection
    # To avoid embedding conflicts (e.g. if it was OpenAI), let's recreate it
    try:
        client.delete_collection(name="product_collection")
    except:
        pass # If it doesn't exist, ignore
        
    collection = client.create_collection(
        name="product_collection",
        embedding_function=emb_fn
    )

    # 5. Prepare Data for Chroma
    # We'll use the 'Description' and 'Title' as the text to embed
    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        # Combine title and description for better search context
        combined_text = f"Title: {row['Title']}\nDescription: {row['Description']}"
        documents.append(combined_text)
        
        # Store other useful info in metadata
        metadatas.append({
            "title": str(row['Title']),
            "price": str(row['Price']),
            "url": str(row['URL']),
            "image_url": str(row['Image URL']),
            "category": str(row['Category'])
        })
        
        # Use ID from CSV or index
        ids.append(str(row['ID']) if row['ID'] else str(index))

    # 6. Add to Collection in batches (Chroma handles large batches well, but let's be safe)
    print(f"Adding {len(documents)} products to ChromaDB...")
    
    # Add in batches of 100 to avoid any potential limits
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        print(f"Imported batch {i//batch_size + 1}")

    print("Import completed successfully!")

if __name__ == "__main__":
    import_products()
