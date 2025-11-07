import os
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Add it to your .env file.")

genai.configure(api_key=GEMINI_KEY)

INPUT_PATH = "cinebrain/data/movies_features.parquet"
OUTPUT_PATH = "cinebrain/data/movies_embeddings.parquet"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _is_vector_ok(vec):
    """Return True if vec looks like a numeric embedding."""
    if isinstance(vec, np.ndarray):
        return vec.dtype.kind in ("f", "i") and vec.size > 0
    if isinstance(vec, (list, tuple)) and len(vec) > 0:
        # ensure it's all numbers
        return all(isinstance(x, (int, float, np.floating, np.integer)) for x in vec)
    return False

def _extract_vector(resp):
    """
    Normalize different Gemini SDK response formats into a plain list[float].
    Supports:
      - resp.embedding.values
      - resp.embedding (list)
      - dict forms with 'embedding' or 'embeddings'
    """
    # Object-like response
    if hasattr(resp, "embedding"):
        emb = resp.embedding
        # google SDK often exposes `values`
        if hasattr(emb, "values"):
            return list(emb.values)
        # sometimes it is already a list
        if isinstance(emb, (list, tuple, np.ndarray)):
            return list(emb)

    # Dict-like response
    if isinstance(resp, dict):
        if "embedding" in resp and isinstance(resp["embedding"], (list, tuple)):
            return list(resp["embedding"])
        if "embedding" in resp and hasattr(resp["embedding"], "values"):
            return list(resp["embedding"].values)
        # Some SDKs return {"embeddings": [{"values": [...]} , ...]}
        if "embeddings" in resp and isinstance(resp["embeddings"], list):
            first = resp["embeddings"][0]
            if isinstance(first, dict) and "values" in first:
                return list(first["values"])

    # Nothing matched
    return None

def embed_one(text, model="text-embedding-004", retries=3, pause=0.6):
    """
    Embed a single text robustly. Retries on transient errors.
    Returns list[float] or None.
    """
    text = (text or "").strip()
    if not text:
        return None

    for attempt in range(1, retries + 1):
        try:
            resp = genai.embed_content(model=model, content=text)
            vec = _extract_vector(resp)
            if _is_vector_ok(vec):
                return vec
            else:
                # Some SDK versions put vector directly on resp (rare)
                if _is_vector_ok(resp):
                    return list(resp)
                # Fall through to retry/log
                print(f"⚠️ Unexpected embedding shape on attempt {attempt}: {type(resp)} -> {str(resp)[:120]}...")
        except Exception as e:
            print(f"⚠️ Embed error (attempt {attempt}): {e}")
        time.sleep(pause * attempt)  # backoff

    return None

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"❌ {INPUT_PATH} not found. Run features.py first to create movies_features.parquet."
        )

    df = pd.read_parquet(INPUT_PATH)
    if "text_blob" not in df.columns:
        raise ValueError("❌ 'text_blob' column missing. Re-run features.py.")

    print(f"🔹 Loaded {len(df)} movies for embedding generation")

    embeddings = []
    for text in tqdm(df["text_blob"].tolist(), desc="Generating embeddings (1-by-1)"):
        vec = embed_one(text)
        embeddings.append(vec)

    df["embedding"] = embeddings

    # Keep only rows with valid vectors
    before = len(df)
    df = df[df["embedding"].apply(_is_vector_ok)].reset_index(drop=True)
    after = len(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"✅ Saved {after} / {before} embeddings → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
