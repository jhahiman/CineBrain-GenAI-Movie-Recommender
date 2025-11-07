# CineBrain-GenAI-Movie-Recommender
# 🎬 CineBrain – GenAI Movie Recommender

> 🚀 A Generative-AI powered movie recommendation engine that combines **real-time TMDb data**, **Gemini embeddings**, **FAISS vector search**, and an interactive **Streamlit UI**.

---

### 🌟 Overview

CineBrain is a next-generation **AI Movie Recommendation System** that recommends movies based on **natural-language descriptions** like:

> _"mind-bending sci-fi with emotional depth"_  
> _"a romantic comedy set in New York"_  

Unlike traditional keyword filters, CineBrain uses **semantic embeddings from Google Gemini** and **vector similarity search** to understand intent and recommend movies that *feel right* — not just sound similar.

---

### 🧠 Architecture

```text
                ┌────────────────────┐
                │   TMDb API (Live)  │
                └────────┬───────────┘
                         │
             (1) Fetch raw movies → JSON
                         │
               ▼
        ┌────────────────────────┐
        │  ETL Layer (fetch_tmdb)│
        │  ▸ movies.parquet      │
        │  ▸ movies.duckdb       │
        └────────┬───────────────┘
                 │
     (2) Feature Engineering
                 │
                 ▼
        ┌────────────────────────┐
        │ models/features.py     │
        │ ▸ text_blob generation │
        └────────┬───────────────┘
                 │
   (3) Embedding Generation (Gemini)
                 │
                 ▼
        ┌────────────────────────┐
        │ models/embeddings.py   │
        │ ▸ text-embedding-004   │
        └────────┬───────────────┘
                 │
    (4) Vector Index (FAISS cosine)
                 │
                 ▼
        ┌────────────────────────┐
        │ models/vector_index.py │
        │ ▸ faiss_index.bin      │
        │ ▸ faiss_index_map.parq │
        └────────┬───────────────┘
                 │
     (5) Streamlit App (GenAI UI)
                 │
                 ▼
        ┌────────────────────────┐
        │ app/streamlit_app.py   │
        │ ▸ Gemini explanations  │
        │ ▸ Posters & metadata   │
        └────────────────────────┘
