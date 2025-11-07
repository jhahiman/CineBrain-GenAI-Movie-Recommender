import os
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

INDEX_PATH = "cinebrain/data/faiss_index.bin"
MAP_PATH = "cinebrain/data/faiss_index_map.parquet"

def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(MAP_PATH):
        raise FileNotFoundError("❌ Run vector_index.py first.")
    index = faiss.read_index(INDEX_PATH)
    mapping = pd.read_parquet(MAP_PATH)
    return index, mapping

def embed_query(text, model="text-embedding-004"):
    """Get embedding for a user query or movie title."""
    try:
        resp = genai.embed_content(model=model, content=text)
        if hasattr(resp, "embedding"):
            return np.array(resp.embedding.values, dtype="float32")
        elif isinstance(resp, dict) and "embedding" in resp:
            return np.array(resp["embedding"], dtype="float32")
    except Exception as e:
        print("⚠️ Embed query failed:", e)
    return None

def recommend_by_text(query, k=5):
    """
    Recommend movies similar to a natural language description or another movie title.
    """
    index, mapping = load_index()
    qvec = embed_query(query)
    if qvec is None:
        print("❌ Could not embed query.")
        return None

    # normalize for cosine similarity
    qvec = qvec.reshape(1, -1)
    faiss.normalize_L2(qvec)

    D, I = index.search(qvec, k)
    results = mapping.iloc[I[0]].copy()
    results["similarity"] = D[0]
    return results[["title", "year", "genres", "director", "similarity"]]

if __name__ == "__main__":
    # Example test
    query = "mind-bending sci-fi with deep emotional story like Inception"
    recs = recommend_by_text(query, k=5)
    print("🔍 Recommendations for:", query)
    print(recs)
