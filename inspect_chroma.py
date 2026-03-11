
import chromadb.utils.embedding_functions as ef
import inspect

def inspect_ef(obj):
    print(f"Inspecting {type(obj).__name__}:")
    for name, value in inspect.getmembers(obj):
        if not name.startswith("__") or name == "__call__":
            try:
                print(f"  {name}: {value}")
            except:
                pass

print("DefaultEmbeddingFunction:")
inspect_ef(ef.DefaultEmbeddingFunction())

if hasattr(ef, "OpenAIEmbeddingFunction"):
    print("\nOpenAIEmbeddingFunction:")
    # We won't instantiate it as it needs a key, but we'll check the class
    for name, value in inspect.getmembers(ef.OpenAIEmbeddingFunction):
        if not name.startswith("__") or name == "__call__":
             print(f"  {name}: {value}")
