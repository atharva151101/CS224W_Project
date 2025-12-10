import numpy as np
import torch
from torch_geometric.data import HeteroData
from pathlib import Path

PROCESSED = Path("imdb-hetero/data/processed")


required_files = [
    "acted_in_edge_index.npz",
    "directed_edge_index.npz",
    "movie_features.npz",
    "person_features.npz"
]

missing_files = [f for f in required_files if not (PROCESSED / f).exists()]
if missing_files:
    raise FileNotFoundError(
        f"Missing files: {missing_files}"
    )

acted = np.load(PROCESSED / "acted_in_edge_index.npz")["edge_index"]
directed = np.load(PROCESSED / "directed_edge_index.npz")["edge_index"]
print(f"  Acted_in edges: {acted.shape[1]}")
print(f"  Directed edges: {directed.shape[1]}")

movie_features_path = PROCESSED / "movie_features_with_stats.npz"
person_features_path = PROCESSED / "person_features_with_stats.npz"

if movie_features_path.exists() and person_features_path.exists():
    movie_x = np.load(movie_features_path)["x"]
    person_x = np.load(person_features_path)["x"]
    print(f"  Movie features: {movie_x.shape}")
    print(f"  Person features: {person_x.shape}")
else:
    if not (PROCESSED / "movie_features.npz").exists():
        raise FileNotFoundError(f"Movie features not found at {PROCESSED / 'movie_features.npz'}")
    if not (PROCESSED / "person_features.npz").exists():
        raise FileNotFoundError(f"Person features not found at {PROCESSED / 'person_features.npz'}")
    
    movie_x = np.load(PROCESSED / "movie_features.npz")["x"]
    person_x = np.load(PROCESSED / "person_features.npz")["x"]
    print(f"  Movie features: {movie_x.shape}")
    print(f"  Person features: {person_x.shape}")

data = HeteroData()
data["movie"].x = torch.tensor(movie_x, dtype=torch.float32)
data["person"].x = torch.tensor(person_x, dtype=torch.float32)
data[("person", "acted_in", "movie")].edge_index = torch.tensor(acted, dtype=torch.long)
data[("person", "directed", "movie")].edge_index = torch.tensor(directed, dtype=torch.long)
data[("movie", "rev_acted_in", "person")].edge_index = data[("person", "acted_in", "movie")].edge_index.flip(0)
data[("movie", "rev_directed", "person")].edge_index = data[("person", "directed", "movie")].edge_index.flip(0)

data["movie"].y = data["movie"].x[:, 1]

output_path = PROCESSED / "imdb_hetero_graph.pt"
torch.save(data, output_path)
print(f"\nSaved graph to: {output_path}")
print(f"  Edges:")
for rel, store in data.edge_items():
    print(f"    {rel}: {tuple(store.edge_index.shape)}")
