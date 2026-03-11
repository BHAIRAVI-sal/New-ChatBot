import chromadb
import os

client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = client.get_collection(name="product_collection")
    count = collection.count()
    print(f"Collection 'product_collection' has {count} documents.")
except Exception as e:
    print(f"Error accessing collection: {e}")
