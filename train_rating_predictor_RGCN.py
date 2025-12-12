import argparse
import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import add_safe_globals
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv

# The saved graph uses pickle, so register HeteroData as a safe global.
add_safe_globals([HeteroData])


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_node_features(data: HeteroData) -> None:
    """
    z-score normalization per node type.
    """
    for ntype in data.node_types:
        x = data[ntype].x
        if x is None:
            continue
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
        data[ntype].x = (x - mean) / std


class RGCNEncoder(nn.Module):
    def __init__(
        self,
        metadata,
        in_channels: Dict[str, int],
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        # initial projection to project features to a common hidden_dim size
        self.proj = nn.ModuleDict()
        self.ln = nn.ModuleDict()
        for ntype, in_dim in in_channels.items():
            self.proj[ntype] = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.ln[ntype] = nn.LayerNorm(hidden_channels)


        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            # Create a SAGEConv operator for each relation type
            for src, rel, dst in metadata[1]:
                conv_dict[(src, rel, dst)] = SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                    aggr='mean'
                )
            # Build the HeteroCOnv operator
            conv = HeteroConv(conv_dict, aggr="mean")
            self.convs.append(conv)


    def forward(self, x_dict, edge_index_dict):
        # initial projection + dropout
        x_dict = {
            ntype: F.dropout(self.proj[ntype](x), p=self.dropout, training=self.training)
            for ntype, x in x_dict.items()
        }


        for i, conv in enumerate(self.convs):
            out_dict = conv(x_dict, edge_index_dict)
            # Apply dropout 
            out_dict = {
                ntype: F.dropout(out, p=self.dropout, training=self.training)
                for ntype, out in out_dict.items()
            }


            new_x = {}
            for ntype, out in out_dict.items():
            # Add skip connections
                new_x[ntype] = self.ln[ntype](x_dict[ntype] + out)


            # Activation except for last layer
            if i < len(self.convs) - 1:
                new_x = {ntype: F.relu(x) for ntype, x in new_x.items()}


            x_dict = new_x


        return x_dict


class RatingPredictor(nn.Module):
    """
    Regression head to predict movie ratings from movie node embeddings.
    """
    
    def __init__(self, hidden_channels: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, movie_embeddings: torch.Tensor) -> torch.Tensor:
        return self.mlp(movie_embeddings).squeeze(-1)


def build_message_passing_edges(data: HeteroData) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """
    Build an edge_index_dict for message passing.
    """
    edge_dict = {}
    for etype in data.edge_types:
        edge_dict[etype] = data[etype].edge_index
    return edge_dict


@torch.no_grad()
def evaluate(
    encoder: RGCNEncoder,
    predictor: RatingPredictor,
    data: HeteroData,
    split: str = "val",
    filter_missing: bool = True,
) -> Tuple[float, float, float]:
    """
    Evaluate rating prediction with MSE, MAE, and RMSE.
    """
    encoder.eval()
    predictor.eval()
    
    mp_edges = build_message_passing_edges(data)
    z_dict = encoder(data.x_dict, mp_edges)
    
    movie_embeddings = z_dict["movie"]
    true_ratings = data["movie"].y
    
    if split == "train":
        mask = data["movie"].train_mask
    elif split == "val":
        mask = data["movie"].val_mask
    elif split == "test":
        mask = data["movie"].test_mask
    else:
        raise ValueError(f"Unknown split: {split}")
    
    split_indices = mask.nonzero(as_tuple=True)[0]
    
    if filter_missing:
        valid_rating_mask = (true_ratings[split_indices] >= 0.0)  # Ratings >= 0 are valid
        split_indices = split_indices[valid_rating_mask]
    
    if len(split_indices) == 0:
        return 0.0, 0.0, 0.0
    
    pred_ratings = predictor(movie_embeddings[split_indices])
    true_ratings_split = true_ratings[split_indices]
    
    mse = F.mse_loss(pred_ratings, true_ratings_split).item()
    mae = F.l1_loss(pred_ratings, true_ratings_split).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    
    return mse, mae, rmse


def print_graph_details(data: HeteroData) -> None:
    """Print information about the graph."""
    print("\n" + "="*80)
    print("GRAPH DETAILS")
    print("="*80)
    
    if "person" in data.node_types:
        num_people = data["person"].num_nodes
        person_feat_dim = data["person"].x.size(-1) if data["person"].x is not None else 0
        print(f"Number of person nodes: {num_people}")
        print(f"Person node feature dimension: {person_feat_dim}")
    
    if "movie" in data.node_types:
        num_movies = data["movie"].num_nodes
        movie_feat_dim = data["movie"].x.size(-1) if data["movie"].x is not None else 0
        print(f"Number of movie nodes: {num_movies}")
        print(f"Movie node feature dimension: {movie_feat_dim}")
        
        if hasattr(data["movie"], "train_mask"):
            train_count = data["movie"].train_mask.sum().item()
            val_count = data["movie"].val_mask.sum().item()
            test_count = data["movie"].test_mask.sum().item()
            print(f"Movie node splits: Train={train_count}, Val={val_count}, Test={test_count}")
        
        if hasattr(data["movie"], "y") and data["movie"].y is not None:
            ratings = data["movie"].y
            valid_ratings = ratings[ratings >= 0.0]
            if len(valid_ratings) > 0:
                print(f"Movie ratings: {len(valid_ratings)} valid, {len(ratings) - len(valid_ratings)} missing")
                print(f"  Rating range: [{valid_ratings.min().item():.2f}, {valid_ratings.max().item():.2f}]")
                print(f"  Rating mean: {valid_ratings.mean().item():.2f}, std: {valid_ratings.std().item():.2f}")
    
    print()
    
    for edge_type in data.edge_types:
        src, rel_name, dst = edge_type
        if rel_name in ["acted_in", "directed", "rev_acted_in", "rev_directed"]:
            edge_store = data[edge_type]
            total_edges = edge_store.edge_index.size(1) if edge_store.edge_index is not None else 0
            print(f"{rel_name} edges ({src} -> {dst}): {total_edges}")
    
    print("="*80 + "\n")


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    data: HeteroData = torch.load(args.data_path, weights_only=False, map_location=device)

    if (not hasattr(data["movie"], "y")) or data["movie"].y is None:
        if data["movie"].x is None or data["movie"].x.size(1) <= 1:
            raise ValueError("Movie features must include a rating column at index 1 to set labels.")
        data["movie"].y = data["movie"].x[:, 1].clone()
    elif data["movie"].x is not None and data["movie"].x.size(1) > 1 and not torch.allclose(
        data["movie"].y, data["movie"].x[:, 1]
    ):
        pass

    if data["movie"].x is not None and data["movie"].x.size(1) > 1:
        data["movie"].x[:, 1] = 0.0
    
    if not hasattr(data["movie"], "train_mask"):
        raise ValueError("Graph not splitted "
                        "Run check_graph_and_make_splits.py first.")
    
    if args.normalize_features:
        normalize_node_features(data)
    data = data.to(device)
    
    print_graph_details(data)

    in_channels = {ntype: data[ntype].x.size(-1) for ntype in data.node_types}
    encoder = RGCNEncoder(
        data.metadata(),
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    predictor = RatingPredictor(
        hidden_channels=args.hidden_channels,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_mask = data["movie"].train_mask
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    
    if args.filter_missing:
        valid_rating_mask = (data["movie"].y[train_indices] >= 0.0)
        train_indices = train_indices[valid_rating_mask]
    
    num_train = len(train_indices)
    print(f"Training on {num_train} movies with valid ratings")
    
    best_val_rmse, best_epoch = float('inf'), -1

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        predictor.train()
        optimizer.zero_grad()
        
        mp_edges = build_message_passing_edges(data)
        z_dict = encoder(data.x_dict, mp_edges)
        
        movie_embeddings = z_dict["movie"]
        pred_ratings = predictor(movie_embeddings[train_indices])
        true_ratings = data["movie"].y[train_indices]
        
        if args.loss_type == "mae":
            loss = F.l1_loss(pred_ratings, true_ratings)
        else:  
            loss = F.mse_loss(pred_ratings, true_ratings)
        
        loss.backward()
        optimizer.step()
        
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            train_mse, train_mae, train_rmse = evaluate(
                encoder, predictor, data, split="train", filter_missing=args.filter_missing
            )
            val_mse, val_mae, val_rmse = evaluate(
                encoder, predictor, data, split="val", filter_missing=args.filter_missing
            )
            
            is_best = val_rmse < best_val_rmse
            if is_best:
                best_val_rmse, best_epoch = val_rmse, epoch
                best_state = {
                    "encoder": encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "epoch": epoch,
                    "train_mse": train_mse,
                    "train_mae": train_mae,
                    "train_rmse": train_rmse,
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                }
            
            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                f"Train MSE {train_mse:.4f} | Train MAE {train_mae:.4f} | Train RMSE {train_rmse:.4f} | "
                f"Val MSE {val_mse:.4f} | Val MAE {val_mae:.4f} | Val RMSE {val_rmse:.4f}"
                f"{' <- best' if is_best else ''}"
            )

    if best_epoch != -1:
        encoder.load_state_dict(best_state["encoder"])
        predictor.load_state_dict(best_state["predictor"])
        
        test_mse, test_mae, test_rmse = evaluate(
            encoder, predictor, data, split="test", filter_missing=args.filter_missing
        )
        
        print("\n" + "="*80)
        print("FINAL TEST RESULTS:")
        print("="*80)
        print(
            f"Best epoch: {best_epoch} | "
            f"Train RMSE: {best_state['train_rmse']:.4f} | "
            f"Val RMSE: {best_state['val_rmse']:.4f} | "
            f"Test RMSE: {test_rmse:.4f}"
        )
        print(
            f"Train MSE: {best_state['train_mse']:.4f} | "
            f"Val MSE: {best_state['val_mse']:.4f} | "
            f"Test MSE: {test_mse:.4f}"
        )
        print(
            f"Train MAE: {best_state['train_mae']:.4f} | "
            f"Val MAE: {best_state['val_mae']:.4f} | "
            f"Test MAE: {test_mae:.4f}"
        )
        print("="*80)
        
        if args.save_model:
            import os
            os.makedirs(os.path.dirname(args.model_path) if os.path.dirname(args.model_path) else ".", exist_ok=True)
            torch.save(best_state, args.model_path)
            print(f"\nModel saved to: {args.model_path}")
        
        if args.save_predictions:
            import os
            import pandas as pd
            
            encoder.eval()
            predictor.eval()
            mp_edges = build_message_passing_edges(data)
            z_dict = encoder(data.x_dict, mp_edges)
            movie_embeddings = z_dict["movie"]
            true_ratings = data["movie"].y
            test_indices = data["movie"].test_mask.nonzero(as_tuple=True)[0]
            if args.filter_missing:
                valid_mask = (true_ratings[test_indices] >= 0.0)
                test_indices = test_indices[valid_mask]
            pred_ratings = predictor(movie_embeddings[test_indices])
            true_ratings_test = true_ratings[test_indices]
            
            df = pd.DataFrame({
                "movie_index": test_indices.cpu().numpy(),
                "true_rating": true_ratings_test.cpu().detach().numpy(),
                "predicted_rating": pred_ratings.cpu().detach().numpy(),
            })
            os.makedirs(os.path.dirname(args.predictions_path) if os.path.dirname(args.predictions_path) else ".", exist_ok=True)
            df.to_csv(args.predictions_path, index=False)
            print(f"\nTest predictions saved to: {args.predictions_path}")
    else:
        print("Training finished without validation improvements.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RGCN model for movie rating prediction (node regression)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="imdb-hetero/data/processed/imdb_hetero_graph_splits.pt"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["mse", "mae"],
        default="mse"
    )
    parser.add_argument(
        "--normalize_features",
        action="store_true"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument(
        "--no_filter_missing",
        dest="filter_missing",
        action="store_false"
    )
    parser.set_defaults(filter_missing=True)
    parser.add_argument(
        "--no_save_model",
        dest="save_model",
        action="store_false"
    )
    parser.set_defaults(save_model=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default="imdb-hetero/data/processed/best_rating_predictor.pt"
    )
    parser.add_argument(
        "--no_save_predictions",
        dest="save_predictions",
        action="store_false"
    )
    parser.set_defaults(save_predictions=True)
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="imdb-hetero/data/processed/predictions.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

