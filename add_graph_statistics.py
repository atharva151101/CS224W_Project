"""
Adds graph-based statistics to node features.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from torch_geometric.data import HeteroData
from torch.serialization import add_safe_globals

add_safe_globals([HeteroData])

def compute_person_statistics(acted_edges, directed_edges, num_people, num_movies, movie_features):
    """
    Compute graph statistics for person nodes.
    """
    
    stats = np.zeros((num_people, 0), dtype=np.float32)
    
    #Degree statistics
    acted_out_degree = np.zeros(num_people, dtype=np.float32)
    directed_out_degree = np.zeros(num_people, dtype=np.float32)
    
    if acted_edges.size(1) > 0:
        person_indices = acted_edges[0].cpu().numpy()
        unique, counts = np.unique(person_indices, return_counts=True)
        acted_out_degree[unique] = counts.astype(np.float32)
    
    if directed_edges.size(1) > 0:
        person_indices = directed_edges[0].cpu().numpy()
        unique, counts = np.unique(person_indices, return_counts=True)
        directed_out_degree[unique] = counts.astype(np.float32)
    
    total_degree = acted_out_degree + directed_out_degree
    
    stats = np.column_stack([
        stats,
        np.log1p(acted_out_degree),      
        np.log1p(directed_out_degree),
        np.log1p(total_degree),
        acted_out_degree / (total_degree + 1e-8),  
    ])
    
    #Diversity metrics
    if acted_edges.size(1) > 0 and movie_features is not None:
        person_to_movies = defaultdict(list)
        acted_src = acted_edges[0].cpu().numpy()
        acted_dst = acted_edges[1].cpu().numpy()
        
        for person_idx, movie_idx in zip(acted_src, acted_dst):
            person_to_movies[person_idx].append(movie_idx)
            
        # basic(4) + synopsis(768) + keywords(768) + isAdult(1) = 1541
        genre_start_idx = 4 + 768 + 768 + 1 

        if genre_start_idx < movie_features.shape[1]:
            person_genre_diversity = np.zeros(num_people, dtype=np.float32)
            
            if isinstance(movie_features, torch.Tensor):
                movie_features_np = movie_features.cpu().numpy()
            else:
                movie_features_np = movie_features
                
            for person_idx, movie_indices in person_to_movies.items():
                if len(movie_indices) > 0:
                    person_movies = movie_features_np[movie_indices]
                    genre_features = person_movies[:, genre_start_idx:]
                    unique_genres = (genre_features.sum(axis=0) > 0).sum()
                    person_genre_diversity[person_idx] = unique_genres
            
            stats = np.column_stack([stats, person_genre_diversity])
    
    return stats


def compute_movie_statistics(acted_edges, directed_edges, num_movies, num_people, person_features):
    """
    Compute graph statistics for movie nodes.
    """
    stats = np.zeros((num_movies, 0), dtype=np.float32)
    
    #Degree statistics
    acted_in_degree = np.zeros(num_movies, dtype=np.float32)
    directed_in_degree = np.zeros(num_movies, dtype=np.float32)
    
    if acted_edges.size(1) > 0:
        movie_indices = acted_edges[1].cpu().numpy()
        unique, counts = np.unique(movie_indices, return_counts=True)
        acted_in_degree[unique] = counts.astype(np.float32)
    
    if directed_edges.size(1) > 0:
        movie_indices = directed_edges[1].cpu().numpy()
        unique, counts = np.unique(movie_indices, return_counts=True)
        directed_in_degree[unique] = counts.astype(np.float32)
    
    total_degree = acted_in_degree + directed_in_degree
    
    stats = np.column_stack([
        stats,
        np.log1p(acted_in_degree),
        np.log1p(directed_in_degree),
        np.log1p(total_degree),
        acted_in_degree / (total_degree + 1e-8),  
    ])
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Add graph statistics to node features"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="imdb-hetero/data/processed"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="imdb_hetero_graph_splits.pt"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="imdb_hetero_graph_with_stats.pt"
    )
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    input_path = processed_dir / args.input_file
    output_path = processed_dir / args.output_file
    
    print(f"Loading graph from {input_path}...")
    data = torch.load(input_path, weights_only=False)
    
    # Get features
    movie_features = data["movie"].x
    person_features = data["person"].x
    
    num_movies = data["movie"].num_nodes
    num_people = data["person"].num_nodes
    
    print(f"Movies: {num_movies}, People: {num_people}")
    print(f"Original movie features: {movie_features.shape[1]}")
    print(f"Original person features: {person_features.shape[1]}")
    
    
    if ("person", "acted_in", "movie") in data.edge_types:
        acted_store = data[("person", "acted_in", "movie")]
        if hasattr(acted_store, "train_edge_index"):
            acted_edges = acted_store.train_edge_index
        else:
            acted_edges = acted_store.edge_index
    else:
        acted_edges = torch.empty((2, 0), dtype=torch.long)
        
    if ("person", "directed", "movie") in data.edge_types:
        directed_store = data[("person", "directed", "movie")]
        if hasattr(directed_store, "train_edge_index"):
            directed_edges = directed_store.train_edge_index
        else:
            directed_edges = directed_store.edge_index
    else:
        directed_edges = torch.empty((2, 0), dtype=torch.long)
    

    person_stats = compute_person_statistics(
        acted_edges, directed_edges, num_people, num_movies, movie_features
    )
    print(f"Person statistics shape: {person_stats.shape}")
    
    movie_stats = compute_movie_statistics(
        acted_edges, directed_edges, num_movies, num_people, person_features
    )
    print(f"Movie statistics shape: {movie_stats.shape}")
    
    person_stats_tensor = torch.from_numpy(person_stats).to(person_features.device)
    movie_stats_tensor = torch.from_numpy(movie_stats).to(movie_features.device)
    
    data["person"].x = torch.cat([person_features, person_stats_tensor], dim=1)
    data["movie"].x = torch.cat([movie_features, movie_stats_tensor], dim=1)
    
    print(f"New person features: {data['person'].x.shape}")
    print(f"New movie features: {data['movie'].x.shape}")
    
    print(f"\nSaving updated graph to {output_path}...")
    torch.save(data, output_path)


if __name__ == "__main__":
    main()
