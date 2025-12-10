import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="imdb-hetero/data/raw")
    ap.add_argument("--out_dir", type=str, default="imdb-hetero/data/processed")
    ap.add_argument("--plot_encoded_json", type=str, default="imdb-hetero/data/raw/plot_data_encoded.json", help="Path to plot_data_encoded.json")
    args, _ = ap.parse_known_args()
    RAW = Path(args.raw_dir)
    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    movies = pd.read_csv(RAW / "movies.csv")
    people = pd.read_csv(RAW / "people.csv")
    edges  = pd.read_csv(RAW / "actor_director_movies_edges.csv")
    
    plot_json_path = Path(args.plot_encoded_json)
    if not plot_json_path.is_absolute():
        if (RAW / plot_json_path.name).exists():
            plot_json_path = RAW / plot_json_path.name
    
    plot_encodings = {}
    encoding_dim = 768  
    if plot_json_path.exists():
        print(f"Loading plot encodings from: {plot_json_path}")
        with open(plot_json_path, 'r', encoding='utf-8') as f:
            plot_data = json.load(f)
        for entry in plot_data:
            tconst = str(entry.get('tconst', ''))
            if not tconst:
                continue
            synopsis_enc = np.array(entry.get('synopsis_encoding', []), dtype=np.float32)
            keywords_enc = np.array(entry.get('keywords_encoding', []), dtype=np.float32)
            if len(synopsis_enc) != encoding_dim:
                print(f"lensynopsis_enc: {len(synopsis_enc)}", f"encoding_dim: {encoding_dim}")
                synopsis_enc = np.zeros(encoding_dim, dtype=np.float32)
                print(f"Warning: synopsis encoding for {tconst} has wrong dimension, using zero vector")
            if len(keywords_enc) != encoding_dim:
                print(f"len(keywords_enc): {len(keywords_enc)}", f"encoding_dim: {encoding_dim}")
                keywords_enc = np.zeros(encoding_dim, dtype=np.float32)
                print(f"Warning: keywords encoding for {tconst} has wrong dimension, using zero vector")
            plot_encodings[tconst] = {
                'synopsis_encoding': synopsis_enc,
                'keywords_encoding': keywords_enc
            }
        print(f"Loaded encodings for {len(plot_encodings)} movies")
        
        movies["tconst"] = movies["tconst"].astype(str)
        before = len(movies)
        plot_tconsts = pd.DataFrame({"tconst": list(plot_encodings.keys())})
        plot_tconsts["tconst"] = plot_tconsts["tconst"].astype(str)
        movies = movies.merge(plot_tconsts[["tconst"]].drop_duplicates(), on="tconst", how="inner")
        after = len(movies)
        print(f"Filtered by plot encodings: {before} -> {after}")
    else:
        print(f"Warning: plot_data_encoded.json not found at {plot_json_path}, will use zero vectors for encodings")

    movies["tconst"] = movies["tconst"].astype(str)
    people["nconst"] = people["nconst"].astype(str)
    edges["tconst"]  = edges["tconst"].astype(str)
    edges["nconst"]  = edges["nconst"].astype(str)

    movie_ids  = pd.Index(movies["tconst"].unique())
    person_ids = pd.Index(people["nconst"].unique())
    movie_id_map  = {t: i for i, t in enumerate(movie_ids)}
    person_id_map = {n: i for i, n in enumerate(person_ids)}

    if "category" not in edges.columns:
        raise ValueError("edges must include 'category' (actor/actress/director)")
    edges["category"] = edges["category"].astype(str).str.lower()
    acted_edges    = edges[edges["category"].isin(["actor","actress"])][["nconst","tconst"]].copy()
    directed_edges = edges[edges["category"].eq("director")][["nconst","tconst"]].copy()

    acted_edges    = acted_edges[acted_edges["nconst"].isin(person_id_map) & acted_edges["tconst"].isin(movie_id_map)]
    directed_edges = directed_edges[directed_edges["nconst"].isin(person_id_map) & directed_edges["tconst"].isin(movie_id_map)]

    acted_src    = acted_edges["nconst"].map(person_id_map).to_numpy(dtype=np.int64)
    acted_dst    = acted_edges["tconst"].map(movie_id_map).to_numpy(dtype=np.int64)
    directed_src = directed_edges["nconst"].map(person_id_map).to_numpy(dtype=np.int64)
    directed_dst = directed_edges["tconst"].map(movie_id_map).to_numpy(dtype=np.int64)

    mv = movies.set_index("tconst").reindex(movie_ids)
    def num(col, dtype, fill):
        if col not in mv.columns:
            return np.full((len(mv),), fill, dtype=dtype)
        return pd.to_numeric(mv[col], errors="coerce").fillna(fill).astype(dtype).to_numpy()

    m_year   = num("startYear",    "int64",   -1)
    m_rating = num("averageRating","float32", -1.0)
    m_votes  = num("numVotes",     "int64",   -1)
    m_isadult = num("isAdult",    "int64",   0)
    
    synopsis_encodings = []
    keywords_encodings = []
    zero_encoding = np.zeros(encoding_dim, dtype=np.float32)
    
    for tconst in movie_ids:
        if tconst in plot_encodings:
            synopsis_encodings.append(plot_encodings[tconst]['synopsis_encoding'])
            keywords_encodings.append(plot_encodings[tconst]['keywords_encoding'])
        else:
            synopsis_encodings.append(zero_encoding)
            keywords_encodings.append(zero_encoding)
    
    synopsis_encodings = np.array(synopsis_encodings, dtype=np.float32)
    keywords_encodings = np.array(keywords_encodings, dtype=np.float32)
    
    # One-hot encoding for genres
    genres_list = []
    for tconst in movie_ids:
        genres_str = mv.loc[tconst, 'genres'] if 'genres' in mv.columns else ''
        if pd.isna(genres_str) or genres_str == '' or genres_str == '\\N':
            genres_list.append([])
        else:
            genres_list.append([g.strip() for g in str(genres_str).split(',')])
    
    mlb = MultiLabelBinarizer()
    genre_one_hot = mlb.fit_transform(genres_list).astype(np.float32)
    
    # One-hot encoding for region
    regions = mv['region'].fillna('UNKNOWN').astype(str) if 'region' in mv.columns else pd.Series(['UNKNOWN'] * len(movie_ids), index=movie_ids)
    regions = regions.reindex(movie_ids).fillna('UNKNOWN')
    lb = LabelBinarizer()
    region_one_hot = lb.fit_transform(regions).astype(np.float32)
    if region_one_hot.ndim == 1:
        region_one_hot = region_one_hot.reshape(-1, 1)
    
    movie_x = np.hstack([
        m_year.astype(np.float32).reshape(-1, 1),
        m_rating.astype(np.float32).reshape(-1, 1),
        m_votes.astype(np.float32).reshape(-1, 1),
        synopsis_encodings,
        keywords_encodings,
        m_isadult.astype(np.float32).reshape(-1, 1),
        genre_one_hot,
        region_one_hot
    ])
    
    np.save(OUT / "genre_classes.npy", mlb.classes_)
    np.save(OUT / "region_classes.npy", lb.classes_)

    pv = people.set_index("nconst").reindex(person_ids)
    def num_person(col, dtype, fill):
        if col not in pv.columns:
            return np.full((len(pv),), fill, dtype=dtype)
        return pd.to_numeric(pv[col], errors="coerce").fillna(fill).astype(dtype).to_numpy()
    
    p_birth = num_person("birthYear", "int64", 9999)
    p_death = num_person("deathYear", "int64", 9999)
    
    person_x = np.stack([
        p_birth.astype(np.float32),
        p_death.astype(np.float32)
    ], axis=1)

    np.savez(OUT / "acted_in_edge_index.npz", edge_index=np.vstack([acted_src, acted_dst]))
    np.savez(OUT / "directed_edge_index.npz", edge_index=np.vstack([directed_src, directed_dst]))
    np.savez(OUT / "movie_features.npz", x=movie_x)
    np.savez(OUT / "person_features.npz", x=person_x)
    pd.Series(movie_id_map).rename("movie_idx").to_csv(OUT / "movie_id_map.csv")
    pd.Series(person_id_map).rename("person_idx").to_csv(OUT / "person_id_map.csv")

    print("Movies:", movie_x.shape[0], "People:", person_x.shape[0])
    print("Movie features shape:", movie_x.shape)
    print("  - Basic features (year, rating, votes, isAdult):", 4)
    print("  - Synopsis encoding:", encoding_dim)
    print("  - Keywords encoding:", encoding_dim)
    print("  - Genre one-hot:", genre_one_hot.shape[1])
    print("  - Region one-hot:", region_one_hot.shape[1])
    print("Person features shape:", person_x.shape)
    print("  - Features (birthYear, deathYear):", 2)
    print("Edges acted_in:", acted_src.shape[0], "directed:", directed_src.shape[0])
    print("Artifacts saved to:", OUT.resolve())

if __name__ == "__main__":
    main()
