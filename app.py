from flask import Flask, render_template, request, jsonify
import chromadb
from chromadb.utils import embedding_functions
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize Gemini AI Client
gemini_key = os.getenv("GEMINI_API_KEY")

class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, client, model_name="models/gemini-embedding-001"):
        self.client = client
        self.model_name = model_name

    def __call__(self, input):
        # input is a list of strings
        if isinstance(input, str):
            input = [input]
        try:
            res = self.client.models.embed_content(
                model=self.model_name,
                contents=input
            )
            return [[float(v) for v in e.values] for e in res.embeddings]
        except Exception as e:
            print(f"Embedding error: {e}")
            raise e

    def name(self):
        return self.model_name

if gemini_key:
    print(f"Initializing Gemini Client with key: {gemini_key[:5]}...")
    client_genai = genai.Client(api_key=gemini_key)
    emb_fn = GeminiEmbeddingFunction(client_genai)
else:
    print("No Gemini API key found, using default embeddings.")
    client_genai = None
    emb_fn = embedding_functions.DefaultEmbeddingFunction()

# Initialize ChromaDB Client
print("Connecting to ChromaDB...")
client_chroma = chromadb.PersistentClient(path="./chroma_db")

# Get the collection
try:
    print("Getting or creating collection...")
    collection = client_chroma.get_or_create_collection(
        name="product_collection",
        embedding_function=emb_fn
    )
    print("Collection loaded.")
except Exception as e:
    print(f"Collection initialization error details: {e}")
    import traceback
    traceback.print_exc()
    collection = client_chroma.get_collection(name="product_collection")
    collection._embedding_function = emb_fn

@app.route('/')
def home():
    try:
        count = collection.count()
    except:
        count = 0
    return render_template('index.html', product_count=count)

@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('q', '')
    if len(query) < 2:
        return jsonify([])
    
    # We want to return ONLY existing product names.
    # We can use collection.get to filter by title in metadata if supported,
    # or use query with a small n_results and extract titles.
    # To be precise about "product names that exist", we'll query but focus on titles.
    results = collection.query(
        query_texts=[query],
        n_results=10 # Get a few more to filter/ensure diversity
    )
    
    # Extract titles from metadata and ensure they are unique and match the query loosely
    suggestions = []
    if results['metadatas']:
        for meta in results['metadatas'][0]:
            title = meta.get('title')
            if title and title not in suggestions:
                suggestions.append(title)
                if len(suggestions) >= 5:
                    break
    
    return jsonify(suggestions)

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    try:
        count = collection.count()
    except:
        count = 0
        
    if not query:
        return render_template('index.html', results=[], product_count=count)
    
    # Query ChromaDB for top 6 similar products
    # n_results=6 provides a nice grid layout
    results = collection.query(
        query_texts=[query],
        n_results=6
    )
    
    # Format the results for the template
    formatted_results = []
    for i in range(len(results['documents'][0])):
        formatted_results.append({
            'document': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i],
            'id': results['ids'][0][i]
        })
        
    return render_template('index.html', query=query, results=formatted_results, product_count=count)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    if not client_genai:
        return jsonify({"response": "I'm sorry, my AI core is not configured. Please check the GEMINI_API_KEY in the .env file."})

    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"response": "Please enter a message so I can help you!"})

    # 1. Retrieve context from ChromaDB
    try:
        print(f"Chatbot Search: querying for '{user_message}'")
        results = collection.query(
            query_texts=[user_message],
            n_results=4
        )
        
        if not results['documents'] or not results['documents'][0]:
            print("Chatbot Search: No relevant products found.")
            return jsonify({"response": "I couldn't find relevant products right now, but I'm here to help. Could you try asking in a different way?"})

        context_text = ""
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            title = meta.get('title', 'Unknown Product')
            price = meta.get('price', 'N/A')
            img_url = meta.get('image_url', '')
            context_text += f"\n- {title} (Price: {price}, Image: {img_url}): {doc}"

    except Exception as e:
        print(f"ChromaDB Error: {e}")
        # fallback behavior if search fails
        return jsonify({"response": "I'm having trouble searching our product catalog right now. could you please rephrase your question?"})

    # 2. Construct Prompt
    prompt = f"""You are a precise, direct, and helpful AI Product Assistant for a health and wellness store.
Answer ONLY what is asked. Be extremely concise.

USER QUESTION: "{user_message}"

RELEVANT PRODUCT CONTEXT:
{context_text}

INSTRUCTIONS:
- Directly answer the question using the context.
- If a product is mentioned, include its price and its image in HTML format: <img src="URL" style="width:100%; border-radius:10px; margin-top:10px;">
- DO NOT add extra chatty filler like "I can help with that" or "Here is the info". Just give the info.
- If the question is about price, just state the price.
- Keep it to 1-2 direct sentences.
"""

    # 3. Generate Response
    import time
    max_retries = 2
    retry_delay = 2 # seconds
    
    for attempt in range(max_retries + 1):
        try:
            print(f"Chatbot Gemini: Generating content (attempt {attempt + 1})...")
            response = client_genai.models.generate_content(
                model='gemini-2.0-flash-lite',
                contents=prompt
            )
            return jsonify({"response": response.text.strip()})
        except Exception as e:
            print(f"Gemini Error (attempt {attempt + 1}): {e}")
            error_msg = str(e)
            
            if attempt < max_retries and ("429" in error_msg or "quota" in error_msg.lower()):
                print(f"Quota hit, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
                
            if "429" in error_msg or "quota" in error_msg.lower():
                # Build a RICH fallback list - no quota message
                fallback_items = []
                for i in range(min(len(results['documents'][0]), 2)):
                    meta = results['metadatas'][0][i]
                    title = meta.get('title')
                    price = meta.get('price', 'N/A')
                    img = meta.get('image_url', '')
                    item_html = f"<b>{title}</b>: €{price}"
                    if img:
                        item_html += f'<br><img src="{img}" style="width:100%; border-radius:10px; margin-top:5px;">'
                    fallback_items.append(item_html)
                
                return jsonify({"response": "<br><br>".join(fallback_items)})
            
            break

    # Fallback if Gemini fails completely
    fallback_items = []
    for i in range(min(len(results['documents'][0]), 2)):
        meta = results['metadatas'][0][i]
        title = meta.get('title')
        price = meta.get('price', 'N/A')
        img = meta.get('image_url', '')
        item_html = f"<b>{title}</b>: €{price}"
        if img:
            item_html += f'<br><img src="{img}" style="width:100%; border-radius:10px; margin-top:5px;">'
        fallback_items.append(item_html)
    
    if fallback_items:
        return jsonify({"response": "<br><br>".join(fallback_items)})
    return jsonify({"response": "I'm sorry, I'm having trouble retrieving that information right now."})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
