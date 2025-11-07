import os
import duckdb
import pandas as pd
import re

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def clean_text(text):
    """Basic cleaning: remove line breaks, extra spaces, and punctuation spacing."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    return text


def build_text_blob(row):
    """Combine title, overview, genres, cast, and director into a single string."""
    parts = []
    if row.get("title"):
        parts.append(f"Title: {row['title']}.")
    if row.get("overview"):
        parts.append(row["overview"])
    if row.get("genres"):
        parts.append(f"Genres: {row['genres']}.")
    if row.get("cast"):
        parts.append(f"Top Cast: {row['cast']}.")
    if row.get("director"):
        parts.append(f"Directed by {row['director']}.")
    return " ".join(parts)

# ------------------------------------------------------------
# Main feature generation pipeline
# ------------------------------------------------------------
def main():
    input_path = "cinebrain/data/movies.duckdb"
    output_path = "cinebrain/data/movies_features.parquet"

    if not os.path.exists(input_path):
        raise FileNotFoundError("❌ movies.duckdb not found. Run fetch_tmdb.py first.")

    con = duckdb.connect(input_path)
    df = con.execute("SELECT * FROM movies").fetchdf()
    con.close()

    print(f"🔹 Loaded {len(df)} raw movies from {input_path}")

    # Clean relevant text columns
    for col in ["title", "overview", "genres", "cast", "director"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
        else:
            df[col] = ""

    # Add 'year' column (from release_date)
    df["year"] = df["release_date"].fillna("").apply(
        lambda x: x[:4] if isinstance(x, str) and len(x) >= 4 else None
    )

    # Build text blob (used for embeddings)
    df["text_blob"] = df.apply(build_text_blob, axis=1)

    # Drop rows with insufficient info
    df = df[df["text_blob"].str.len() > 30].reset_index(drop=True)

    # ✅ Preserve poster_path and other metadata
    keep_cols = [
        "id",
        "title",
        "year",
        "release_date",
        "popularity",
        "vote_average",
        "runtime",
        "overview",
        "genres",
        "cast",
        "director",
        "poster_path",   # <-- ✅ Include this
        "text_blob",
    ]
    # keep only columns that exist
    df = df[[c for c in keep_cols if c in df.columns]]

    # Save enriched features
    os.makedirs("cinebrain/data", exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"✅ Saved {len(df)} enriched records with poster_path → {output_path}")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
