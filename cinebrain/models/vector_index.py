import os
import faiss
import numpy as np
import pandas as pd

def build_faiss_index(
    embeddings_path="cinebrain/data/movies_embeddings.parquet",
    index_path="cinebrain/data/faiss_index.bin"
):
    """
    Build a FAISS cosine-similarity index for movie embeddings and save mapping with metadata.
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError("❌ Embeddings file not found. Run embeddings.py first.")

    df = pd.read_parquet(embeddings_path)
    print(f"🔹 Loaded {len(df)} embeddings from {embeddings_path}")

    # Convert embeddings to numpy array (float32)
    X = np.vstack(df["embedding"].to_numpy()).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(X)

    # Build cosine-similarity FAISS index
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    # Save FAISS index
    faiss.write_index(index, index_path)

    # ✅ Include poster_path and other useful metadata
    map_path = index_path.replace(".bin", "_map.parquet")

    keep_cols = [
        "id",
        "title",
        "year",
        "genres",
        "director",
        "poster_path",    # ✅ now included
        "overview",       # ✅ used for Gemini explanation
        "vote_average",   # ✅ optional - can show rating later
        "runtime",        # ✅ optional - for extra info
    ]

    # keep only columns that exist in df
    mapping = df[[c for c in keep_cols if c in df.columns]].copy()
    mapping.to_parquet(map_path, index=False)

    print(f"✅ FAISS index built and saved → {index_path}")
    print(f"✅ Mapping saved → {map_path} with columns: {', '.join(mapping.columns)}")

if __name__ == "__main__":
    build_faiss_index()
