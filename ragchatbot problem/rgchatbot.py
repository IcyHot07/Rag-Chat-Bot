import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os


def create_dataset():
    data = [
        {"id": 1, "customer": "Amit", "product": "Laptop", "amount": 55000, "date": "2024-01-12"},
        {"id": 2, "customer": "Amit", "product": "Mouse", "amount": 700, "date": "2024-02-15"},
        {"id": 3, "customer": "Riya", "product": "Mobile", "amount": 30000, "date": "2024-01-05"},
        {"id": 4, "customer": "Riya", "product": "Earbuds", "amount": 1500, "date": "2024-02-20"},
        {"id": 5, "customer": "Karan", "product": "Keyboard", "amount": 1200, "date": "2024-03-01"}
    ]
    with open('transactions.json', 'w') as f:
        json.dump(data, f, indent=4)
    print("✅ Dataset 'transactions.json' created successfully.")


def load_and_preprocess():
    with open('transactions.json', 'r') as f:
        data = json.load(f)
    
    texts = []
    for item in data:
        desc = f"On {item['date']}, {item['customer']} purchased a {item['product']} for {item['amount']}."
        texts.append(desc)
    
    return texts


def create_embeddings(texts):
    print("⚡ Using lightweight/fake embeddings for testing...")
    embeddings = np.random.rand(len(texts), 384) 
    model = None 
    return model, embeddings


def retrieve_transactions(query, model, embeddings, texts, top_k=3):
    print(f"DEBUG: Searching for '{query}'...")
    
    scores = []
    query_words = query.lower().split()
    
    for text in texts:
        score = sum(1 for word in query_words if word in text.lower())
        scores.append(score)
    
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    return [texts[i] for i in top_indices]
    
    
    results = [texts[i] for i in top_indices]
    return results


def generate_response(query, context_texts):
    context_block = "\n".join(context_texts)
    
    prompt = f"""
    You are a helpful assistant for a retail company.
    Answer the user's question using ONLY the context provided below.
    
    Context:
    {context_block}
    
    Question: {query}
    
    Answer:
    """
    
    try:
        return f"[MOCK LLM OUTPUT]\nI found relevant data:\n{context_block}\n\n(To get a natural language answer like 'Amit spent 55700', please enable the OpenAI API code block in the script.)"
        
    except Exception as e:
        return f"Error generating response: {e}"


if __name__ == "__main__":
    create_dataset()
    
    texts = load_and_preprocess()
    
    model, embeddings = create_embeddings(texts)
    
    print("\n🤖 RAG Chatbot Ready! (Type 'exit' to quit)\n")
    
    while True:
        user_query = input("User: ")
        if user_query.lower() in ['exit', 'quit']:
            break
            
        retrieved_context = retrieve_transactions(user_query, model, embeddings, texts)
        
        response = generate_response(user_query, retrieved_context)
        
        print(f"Bot: {response}\n")