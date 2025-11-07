import duckdb

con = duckdb.connect("cinebrain/data/movies.duckdb")
df = con.execute("SELECT title, genres, director, runtime FROM movies LIMIT 5").fetchdf()
print(df)



import pandas as pd
df = pd.read_parquet("cinebrain/data/movies_features.parquet")
print(df.head(2)[["title", "text_blob"]])
