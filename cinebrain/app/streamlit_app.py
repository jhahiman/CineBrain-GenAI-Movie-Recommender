import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

INDEX_PATH = "cinebrain/data/faiss_index.bin"
MAP_PATH = "cinebrain/data/faiss_index_map.parquet"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def fmt_runtime(minutes):
    """Return runtime like '2h 14m' or '—' if missing."""
    if minutes is None or not isinstance(minutes, (int, float)) or math.isnan(float(minutes)):
        return "—"
    minutes = int(minutes)
    h, m = divmod(minutes, 60)
    if h and m:
        return f"{h}h {m}m"
    if h:
        return f"{h}h"
    return f"{m}m"

def safe_poster_url(poster_path):
    """Build a safe TMDb poster URL or return None."""
    if isinstance(poster_path, str) and poster_path.strip():
        p = poster_path.strip()
        if p.startswith("http"):
            return p
        if p.startswith("/"):
            return f"https://image.tmdb.org/t/p/w500{p}"
    return None

@st.cache_resource
def load_index_and_mapping():
    """Load FAISS index and mapping dataframe (cached)."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(MAP_PATH):
        st.error("FAISS index or mapping not found. Please run vector_index.py first.")
        st.stop()
    index = faiss.read_index(INDEX_PATH)
    mapping = pd.read_parquet(MAP_PATH)
    return index, mapping

def embed_query(text, model="text-embedding-004"):
    """Embed the user query using Gemini embeddings."""
    try:
        resp = genai.embed_content(model=model, content=text)
        if hasattr(resp, "embedding"):
            return np.array(resp.embedding.values, dtype="float32")
        elif isinstance(resp, dict) and "embedding" in resp:
            return np.array(resp["embedding"], dtype="float32")
    except Exception as e:
        st.error(f"Embedding failed: {e}")
    return None

def recommend(query, k=6):
    """Return top-k most similar movies for a given query string."""
    index, mapping = load_index_and_mapping()
    qvec = embed_query(query)
    if qvec is None:
        return pd.DataFrame()
    qvec = qvec.reshape(1, -1)
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, k)
    recs = mapping.iloc[I[0]].copy()
    recs["similarity"] = D[0]
    return recs

def explain_recommendation(user_query, movie_title, overview=None):
    """Use Gemini to explain why a movie matches the user query."""
    try:
        prompt = (
            f"User is looking for: {user_query}\n"
            f"Recommended movie: {movie_title}\n"
            f"Movie description: {overview or 'N/A'}\n\n"
            "Explain in 1–2 short sentences why this movie fits the user's request. "
            "Make it natural, friendly, and insightful."
        )
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        print(f"⚠️ Explanation failed for {movie_title}: {e}")
    return None

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="🎬 CineBrain", page_icon="🎥", layout="wide")

st.title("🎬 CineBrain – GenAI Movie Recommender")
st.write(
    "Ask for any kind of movie! Example: *'mind-bending sci-fi with emotional depth'* "
    "or *'romantic comedy set in New York'*."
)

query = st.text_input("Describe the kind of movie you want:")

if st.button("Recommend 🎥") or query:
    if query.strip():
        with st.spinner("Finding the best matches..."):
            recs = recommend(query)
        if recs.empty:
            st.warning("No recommendations found.")
        else:
            st.subheader("🎯 Top Recommendations")

            cols = st.columns(2)
            for i, (_, row) in enumerate(recs.iterrows()):
                with cols[i % 2]:
                    # Poster image
                    poster_url = safe_poster_url(row.get("poster_path"))
                    if poster_url:
                        st.image(poster_url, width=250)
                    else:
                        st.image("https://via.placeholder.com/250x370?text=No+Poster", width=250)

                    # Metadata
                    year = int(row["year"]) if pd.notna(row.get("year")) else "N/A"
                    rating = row.get("vote_average")
                    rating_txt = f"{float(rating):.1f}" if pd.notna(rating) else "—"
                    runtime_txt = fmt_runtime(row.get("runtime"))

                    st.markdown(
                        f"**{row.get('title','Untitled')}** ({year})  \n"
                        f"*{row.get('genres','') or '—'}*  \n"
                        f"🎬 **Director:** {row.get('director') or 'Unknown'}  \n"
                        f"⭐ **Rating:** {rating_txt}  |  ⏱ **Runtime:** {runtime_txt}  \n"
                        f"🧠 **Similarity:** {row['similarity']:.3f}"
                    )

                    # Gemini-generated explanation
                    explanation = explain_recommendation(query, row.get("title"), row.get("overview"))
                    if explanation:
                        st.markdown(f"💡 *{explanation}*")

                    st.divider()

st.caption("This product uses the TMDb API but is not endorsed or certified by TMDb.")
