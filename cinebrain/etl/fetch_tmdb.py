import os
import time
import requests
import duckdb
import pandas as pd
from dotenv import load_dotenv

# ------------------------------------------------------------
# Load API key
# ------------------------------------------------------------
load_dotenv()
TMDB_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

if not TMDB_KEY:
    raise ValueError("❌ TMDB_API_KEY not found. Please add it to your .env file.")

# ------------------------------------------------------------
# Helper: safe API GET with retries
# ------------------------------------------------------------
def tmdb_get(endpoint: str, params=None):
    """Make a TMDb API request with retries and API key."""
    url = f"{BASE_URL}{endpoint}"
    headers = {"Authorization": f"Bearer {TMDB_KEY}"} if TMDB_KEY.startswith("ey") else {}
    params = params or {}
    if "api_key" not in params and not headers:
        params["api_key"] = TMDB_KEY

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json()
            else:
                print(f"⚠️ TMDb error {r.status_code}: {r.text[:100]}")
        except Exception as e:
            print(f"⚠️ API call failed: {e}")
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch data from {endpoint}")

# ------------------------------------------------------------
# Fetch genres
# ------------------------------------------------------------
def fetch_genres():
    data = tmdb_get("/genre/movie/list")
    return {g["id"]: g["name"] for g in data.get("genres", [])}

# ------------------------------------------------------------
# Fetch base movie list (popular/top-rated)
# ------------------------------------------------------------
def fetch_movies(pages=5, list_type="popular"):
    """Fetch basic movie data (id, title, release, popularity, poster path, etc.)"""
    movies = []
    for page in range(1, pages + 1):
        data = tmdb_get(f"/movie/{list_type}", {"page": page})
        results = data.get("results", [])
        movies.extend(results)
        print(f"✅ Page {page}: {len(results)} movies fetched ({list_type})")
        time.sleep(0.25)
    return movies

# ------------------------------------------------------------
# Fetch detailed info per movie
# ------------------------------------------------------------
def enrich_movie(movie_id):
    """Fetch runtime, genres, overview, cast, director."""
    try:
        detail = tmdb_get(f"/movie/{movie_id}")
        credits = tmdb_get(f"/movie/{movie_id}/credits")
    except Exception as e:
        print(f"⚠️ Failed to enrich {movie_id}: {e}")
        return {}

    cast = [c["name"] for c in credits.get("cast", [])[:5]]
    crew = [c for c in credits.get("crew", []) if c.get("job") in ["Director", "Writer"]]
    director = next((c["name"] for c in crew if c.get("job") == "Director"), None)
    genres = [g["name"] for g in detail.get("genres", [])]

    return {
        "runtime": detail.get("runtime"),
        "genres": ", ".join(genres),
        "cast": ", ".join(cast),
        "director": director,
        "vote_average": detail.get("vote_average"),
        "overview": detail.get("overview"),
    }

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def main(pages=5, list_type="popular"):
    genres_map = fetch_genres()
    movies = fetch_movies(pages=pages, list_type=list_type)
    print(f"🔹 Fetched {len(movies)} base movies — enriching with details...")

    records = []
    for m in movies:
        try:
            extra = enrich_movie(m["id"])
            record = {
                "id": m["id"],
                "title": m.get("title"),
                "release_date": m.get("release_date"),
                "popularity": m.get("popularity"),
                "poster_path": m.get("poster_path") or None,  # ✅ Always included
                "vote_average": extra.get("vote_average"),
                "runtime": extra.get("runtime"),
                "overview": extra.get("overview"),
                "genres": extra.get("genres"),
                "cast": extra.get("cast"),
                "director": extra.get("director"),
            }
            records.append(record)
            time.sleep(0.2)
        except Exception as e:
            print(f"⚠️ Error processing movie {m.get('title')}: {e}")

    df = pd.DataFrame(records)
    os.makedirs("cinebrain/data", exist_ok=True)

    # Save as Parquet
    parquet_path = "cinebrain/data/movies.parquet"
    duckdb_path = "cinebrain/data/movies.duckdb"

    df.to_parquet(parquet_path, index=False)
    print(f"💾 Saved {len(df)} movies → {parquet_path}")

    # ✅ Always recreate DuckDB table from scratch to include poster_path
    con = duckdb.connect(duckdb_path)
    con.execute("DROP TABLE IF EXISTS movies;")  # <-- ensures clean schema
    con.execute("CREATE TABLE movies AS SELECT * FROM df;")
    con.close()

    print(f"✅ Created DuckDB with {len(df)} movies and columns: {', '.join(df.columns)}")

if __name__ == "__main__":
    main(pages=5, list_type="popular")
