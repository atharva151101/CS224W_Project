import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from torch.serialization import add_safe_globals
from torch_geometric.data import HeteroData
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

add_safe_globals([HeteroData])


def compute_laplacian_eigenvectors(
    train_acted_edges, train_directed_edges, num_people, num_movies, 
    num_eigenvectors=8
):
    """
    Compute Laplacian eigenvectors.
    """
    if isinstance(train_acted_edges, torch.Tensor):
        train_acted_edges = train_acted_edges.cpu().numpy()
    if isinstance(train_directed_edges, torch.Tensor):
        train_directed_edges = train_directed_edges.cpu().numpy()
    
    all_edges = []
    if train_acted_edges.size > 0:
        all_edges.append(train_acted_edges)
    if train_directed_edges.size > 0:
        all_edges.append(train_directed_edges)
    
    if len(all_edges) == 0:
        return (
            np.zeros((num_people, num_eigenvectors), dtype=np.float32),
            np.zeros((num_movies, num_eigenvectors), dtype=np.float32)
        )
    
    train_edges = np.hstack(all_edges) if len(all_edges) > 1 else all_edges[0]
    
    total_nodes = num_people + num_movies
    
    person_indices = train_edges[0]
    movie_indices = train_edges[1] + num_people  
    
    row_indices = np.concatenate([person_indices, movie_indices])
    col_indices = np.concatenate([movie_indices, person_indices])
    
    weights = np.ones(len(row_indices), dtype=np.float32)
    adj = coo_matrix(
        (weights, (row_indices, col_indices)),
        shape=(total_nodes, total_nodes)
    ).tocsr()
    
    adj = (adj + adj.T) / 2.0
    adj.data = np.ones_like(adj.data)
    
    # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    degree = np.array(adj.sum(axis=1)).flatten()
    degree[degree == 0] = 1 
    degree_inv_sqrt = diags(1.0 / np.sqrt(degree))
    
    identity = diags(np.ones(total_nodes))
    laplacian = identity - degree_inv_sqrt @ adj @ degree_inv_sqrt
    
    # Compute smallest k+1 eigenvectors
    print(f"  Computing Laplacian eigenvectors on {total_nodes} nodes...")
    try:
        eigenvals, eigenvecs = eigsh(
            laplacian, k=min(num_eigenvectors + 1, total_nodes - 1),
            which='SM', return_eigenvectors=True
        )
        # Sort by eigenvalue and take smallest
        idx = np.argsort(eigenvals.real)
        all_eigenvectors = eigenvecs[:, idx[1:num_eigenvectors+1]].real.astype(np.float32)
        
        all_eigenvectors = all_eigenvectors / (np.linalg.norm(all_eigenvectors, axis=0, keepdims=True) + 1e-8)
        
        person_eigenvectors = all_eigenvectors[:num_people, :]
        movie_eigenvectors = all_eigenvectors[num_people:, :]
        
    except Exception as e:
        print(f"  Warning: {e} while computing eigenvectors")
        person_eigenvectors = np.zeros((num_people, num_eigenvectors), dtype=np.float32)
        movie_eigenvectors = np.zeros((num_movies, num_eigenvectors), dtype=np.float32)
    
    return person_eigenvectors, movie_eigenvectors


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits and optionally add Laplacian eigenvectors"
    )
    parser.add_argument(
        "--add_eigenvectors",
        action="store_true"
    )
    parser.add_argument(
        "--num_eigenvectors",
        type=int,
        default=128
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1
    )
    args = parser.parse_args()
    
    PROCESSED = Path("imdb-hetero/data/processed")
    data = torch.load(PROCESSED / "imdb_hetero_graph.pt", weights_only=False)

    for rel, store in data.edge_items():
        src,_,dst = rel
        ei = store.edge_index
        assert ei[0].min() >= 0 and ei[0].max() < data[src].num_nodes
        assert ei[1].min() >= 0 and ei[1].max() < data[dst].num_nodes
    print("Index bounds OK.")

    # Splits
    def split_edges(ei, train_ratio=0.8, val_ratio=0.1):
        E = ei.size(1)
        perm = torch.randperm(E)
        ntr, nva = int(E*train_ratio), int(E*val_ratio)
        return ei[:, perm[:ntr]], ei[:, perm[ntr:ntr+nva]], ei[:, perm[ntr+nva:]]

    # Create splits for acted_in relation
    acted_rel = ("person","acted_in","movie")
    if acted_rel in data.edge_types:
        tr, va, te = split_edges(data[acted_rel].edge_index, args.train_ratio, args.val_ratio)
        data[acted_rel].train_edge_index, data[acted_rel].val_edge_index, data[acted_rel].test_edge_index = tr, va, te
    
    
    N = data["movie"].x.size(0)
    perm = torch.randperm(N)
    ntr, nva = int(args.train_ratio*N), int(args.val_ratio*N)
    data["movie"].train_mask = torch.zeros(N, dtype=torch.bool).index_fill_(0, perm[:ntr], True)
    data["movie"].val_mask   = torch.zeros(N, dtype=torch.bool).index_fill_(0, perm[ntr:ntr+nva], True)
    data["movie"].test_mask  = torch.zeros(N, dtype=torch.bool).index_fill_(0, perm[ntr+nva:], True)

    print("Splits created.")

    # Compute Laplacian eigenvectors
    if args.add_eigenvectors:
        num_people = data["person"].x.size(0)
        num_movies = data["movie"].x.size(0)
        
        train_acted = data[("person", "acted_in", "movie")].train_edge_index
        train_directed = torch.empty((2, 0), dtype=torch.long)
        if ("person", "directed", "movie") in data.edge_types:
            train_directed = data[("person", "directed", "movie")].edge_index
        
        person_eigenvectors, movie_eigenvectors = compute_laplacian_eigenvectors(
            train_acted, train_directed, num_people, num_movies,
            num_eigenvectors=args.num_eigenvectors
        )
        
        print(f"  Person eigenvectors shape: {person_eigenvectors.shape}")
        print(f"  Movie eigenvectors shape: {movie_eigenvectors.shape}")
        
        # Add eigenvectors to node features
        person_eigen_tensor = torch.from_numpy(person_eigenvectors)
        movie_eigen_tensor = torch.from_numpy(movie_eigenvectors)
        
        data["person"].x = torch.cat([data["person"].x, person_eigen_tensor], dim=1)
        data["movie"].x = torch.cat([data["movie"].x, movie_eigen_tensor], dim=1)
        
        print(f"  Updated person features: {data['person'].x.shape}")
        print(f"  Updated movie features: {data['movie'].x.shape}")

    torch.save(data, PROCESSED / "imdb_hetero_graph_splits.pt")
    print(f"\nSaved splits at {PROCESSED / 'imdb_hetero_graph_splits.pt'}")

if __name__ == "__main__":
    main()
