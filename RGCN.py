import argparse
import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import add_safe_globals
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.utils import negative_sampling



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

add_safe_globals([HeteroData])


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_node_features(data: HeteroData) -> None:
    """
    In-place z-score normalization per node type.
    """
    for ntype in data.node_types:
        x = data[ntype].x
        if x is None:
            continue
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
        data[ntype].x = (x - mean) / std



class MLPLinkPredictor(nn.Module):
    """MLP-based link predictor"""
    
    def __init__(self, hidden_channels: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, z_dict, edge_index, src_type: str, dst_type: str):
        src_z = z_dict[src_type][edge_index[0]]
        dst_z = z_dict[dst_type][edge_index[1]]
        # Concatenate source and destination embeddings
        combined = torch.cat([src_z, dst_z], dim=-1)
        return self.mlp(combined).squeeze(-1)


def get_edge_split(
    data: HeteroData, rel: Tuple[str, str, str], train_ratio: float, val_ratio: float
) -> HeteroData:
    """
    Create edge splits if the Graph object doesn't already contain them. 
    """
    rel_store = data[rel]
    if hasattr(rel_store, "train_edge_index"):
        return data

    edge_index = rel_store.edge_index
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges, device=edge_index.device)
    n_train = int(train_ratio * num_edges)
    n_val = int(val_ratio * num_edges)

    rel_store.train_edge_index = edge_index[:, perm[:n_train]]
    rel_store.val_edge_index = edge_index[:, perm[n_train : n_train + n_val]]
    rel_store.test_edge_index = edge_index[:, perm[n_train + n_val :]]

    rev_rel = find_reverse_relation(data, rel)
    if rev_rel is not None:
        data[rev_rel].train_edge_index = rel_store.train_edge_index.flip(0)
        data[rev_rel].val_edge_index = rel_store.val_edge_index.flip(0)
        data[rev_rel].test_edge_index = rel_store.test_edge_index.flip(0)
    return data


def find_reverse_relation(data: HeteroData, rel: Tuple[str, str, str]):
    src, _, dst = rel
    for candidate in data.edge_types:
        if candidate[0] == dst and candidate[2] == src:
            return candidate
    return None


def build_message_passing_edges(
    data: HeteroData, target_rel: Tuple[str, str, str]
) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """
    Build an edge_index_dict for message passing that masks the target relation and its reverse, to only use training edges.
    """
    rel_store = data[target_rel]
    rev_rel = find_reverse_relation(data, target_rel)
    edge_dict = {}
    for etype in data.edge_types:
        if etype == target_rel and hasattr(rel_store, "train_edge_index"):
            edge_dict[etype] = rel_store.train_edge_index
        elif rev_rel is not None and etype == rev_rel and hasattr(rel_store, "train_edge_index"):
            edge_dict[etype] = rel_store.train_edge_index.flip(0)
        else:
            edge_dict[etype] = data[etype].edge_index
    return edge_dict


def bce_loss(pred_pos: torch.Tensor, pred_neg: torch.Tensor) -> torch.Tensor:
    pos_labels = torch.ones_like(pred_pos)
    neg_labels = torch.zeros_like(pred_neg)
    scores = torch.cat([pred_pos, pred_neg], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)
    return F.binary_cross_entropy_with_logits(scores, labels)


def bpr_loss(pred_pos: torch.Tensor, pred_neg: torch.Tensor, sample_negatives: bool = True) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss implementation
    """
    num_pos = pred_pos.size(0)
    num_neg = pred_neg.size(0)
    
    if sample_negatives and num_neg >= num_pos:
        neg_indices = torch.randint(0, num_neg, (num_pos,), device=pred_neg.device)
        sampled_neg = pred_neg[neg_indices] 
        diff = pred_pos - sampled_neg  
        loss = -F.logsigmoid(diff).mean()
    else:
        pos_expanded = pred_pos.unsqueeze(1)  
        neg_expanded = pred_neg.unsqueeze(0)  
        diff = pos_expanded - neg_expanded  
        loss = -F.logsigmoid(diff).mean()
    
    return loss


def sample_movie_conditioned_negatives(
    edge_index: torch.Tensor,
    num_src_nodes: int,
    neg_per_pos: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Movie-conditioned negative sampling
    """

    pos_src = edge_index[0]  
    pos_dst = edge_index[1]  
    num_pos = pos_src.size(0)

    neg_dst = pos_dst.repeat_interleave(neg_per_pos)  
    num_neg = num_pos * neg_per_pos
    neg_src = torch.randint(0, num_src_nodes, (num_neg,), device=device)

    pos_src_rep = pos_src.repeat_interleave(neg_per_pos)
    mask = neg_src == pos_src_rep
    while mask.any():
        neg_src[mask] = torch.randint(0, num_src_nodes, (int(mask.sum().item()),), device=device)
        mask = neg_src == pos_src_rep

    neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
    return neg_edge_index


@torch.no_grad()
def compute_recall_at_k_and_mrr(
    encoder: RGCNEncoder,
    predictor: MLPLinkPredictor,
    data: HeteroData,
    rel: Tuple[str, str, str],
    edge_index: torch.Tensor,
    k_values: list = [1, 5, 10, 20, 50],
    batch_size: int = 1000,
    top_k_candidates: int = 1000,
) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """
    Compute Recall@K, Precision@K, and MRR for link prediction.
    
    Uses PinSage like two-stage ranking:
    1. Fast dot-product to get top-k candidates
    2. MLP predictor to re-rank those candidates
    """
    encoder.eval()
    predictor.eval()
    mp_edges = build_message_passing_edges(data, rel)
    z_dict = encoder(data.x_dict, mp_edges)

    src, _, dst = rel
    num_pos = edge_index.size(1)
    
    z_src = z_dict[src]
    z_dst = z_dict[dst]
    
    recall_at_k = {k: 0.0 for k in k_values}
    precision_at_k = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    
    for i in range(0, num_pos, batch_size):
        end_idx = min(i + batch_size, num_pos)
        batch_src = edge_index[0, i:end_idx]
        batch_dst = edge_index[1, i:end_idx]

        all_people = torch.arange(z_src.size(0), device=z_src.device)

        for j in range(len(batch_dst)):
            dst_idx = batch_dst[j]
            true_person = batch_src[j].item()

            # Fast dot-product pre-filtering
            movie_embedding = z_dst[dst_idx]  
            dot_scores = torch.matmul(z_src, movie_embedding)  
            
            top_k = min(top_k_candidates, all_people.size(0))
            _, top_k_indices = torch.topk(dot_scores, k=top_k, largest=True)
            top_k_people = all_people[top_k_indices]  
            
            # MLP re-ranking of top-k candidates
            dst_repeat = dst_idx.repeat(top_k_people.size(0))
            candidate_edge_index = torch.stack([top_k_people, dst_repeat], dim=0)
            
            person_scores = predictor(z_dict, candidate_edge_index, src, dst)  
            
            sorted_indices = torch.argsort(person_scores, descending=True)
            ranked_people = top_k_people[sorted_indices]
            
            true_person_mask = (ranked_people == true_person)
            if true_person_mask.any():
                rank = true_person_mask.nonzero(as_tuple=True)[0].item() + 1
            else:
                rank = top_k + 1  
            
            for k in k_values:
                if rank <= k:
                    recall_at_k[k] += 1.0
                    precision_at_k[k] += 1.0 / k
            
            mrr_sum += 1.0 / rank
    
    recall_at_k = {k: v / num_pos for k, v in recall_at_k.items()}
    precision_at_k = {k: v / num_pos for k, v in precision_at_k.items()}
    mrr = mrr_sum / num_pos
    
    return recall_at_k, precision_at_k, mrr


@torch.no_grad()
def compute_local_recall_at_k_and_mrr(
    encoder: RGCNEncoder,
    predictor: MLPLinkPredictor,
    data: HeteroData,
    rel: Tuple[str, str, str],
    edge_index: torch.Tensor,
    k_values: list = [1, 5, 10, 20, 50],
    batch_size: int = 1000,
    local_candidate_pool_size: int = 500,
) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """
    Compute Local Recall@K, Precision@K, and MRR for link prediction.
    Local recall@k ranks within a smaller candidate pool rather than all possible candidates.
    """
    encoder.eval()
    predictor.eval()
    mp_edges = build_message_passing_edges(data, rel)
    z_dict = encoder(data.x_dict, mp_edges)

    src, _, dst = rel
    num_pos = edge_index.size(1)
    num_people = data[src].num_nodes
    
    z_src = z_dict[src]
    
    local_recall_at_k = {k: 0.0 for k in k_values}
    local_precision_at_k = {k: 0.0 for k in k_values}
    local_mrr_sum = 0.0
    
    for i in range(0, num_pos, batch_size):
        end_idx = min(i + batch_size, num_pos)
        batch_src = edge_index[0, i:end_idx]
        batch_dst = edge_index[1, i:end_idx]

        for j in range(len(batch_dst)):
            dst_idx = batch_dst[j]
            true_person = batch_src[j].item()

            num_candidates = min(local_candidate_pool_size - 1, num_people - 1)
            all_people = torch.arange(num_people, device=z_src.device)
            candidate_mask = (all_people != true_person)
            candidate_pool = all_people[candidate_mask]
            
            if candidate_pool.size(0) > num_candidates:
                perm = torch.randperm(candidate_pool.size(0), device=candidate_pool.device)
                sampled_candidates = candidate_pool[perm[:num_candidates]]
            else:
                sampled_candidates = candidate_pool
            
            local_candidates = torch.cat([sampled_candidates, torch.tensor([true_person], device=z_src.device)])
            
            dst_repeat = dst_idx.repeat(local_candidates.size(0))
            candidate_edge_index = torch.stack([local_candidates, dst_repeat], dim=0)
            candidate_scores = predictor(z_dict, candidate_edge_index, src, dst)  # [pool_size]
            
            sorted_indices = torch.argsort(candidate_scores, descending=True)
            ranked_candidates = local_candidates[sorted_indices]
            
            true_person_mask = (ranked_candidates == true_person)
            if true_person_mask.any():
                rank = true_person_mask.nonzero(as_tuple=True)[0].item() + 1
            else:
                rank = len(local_candidates) + 1
            
            for k in k_values:
                if rank <= k:
                    local_recall_at_k[k] += 1.0
                    local_precision_at_k[k] += 1.0 / k
            
            local_mrr_sum += 1.0 / rank
    
    local_recall_at_k = {k: v / num_pos for k, v in local_recall_at_k.items()}
    local_precision_at_k = {k: v / num_pos for k, v in local_precision_at_k.items()}
    local_mrr = local_mrr_sum / num_pos
    
    return local_recall_at_k, local_precision_at_k, local_mrr


@torch.no_grad()
def evaluate(
    encoder: RGCNEncoder,
    predictor: MLPLinkPredictor,
    data: HeteroData,
    rel: Tuple[str, str, str],
    num_neg: int,
    split: str = "val",
    compute_ranking_metrics: bool = True,
    k_values: list = [1, 5, 10, 20, 50],
    top_k_candidates: int = 1000,
    compute_local_metrics: bool = True,
    local_candidate_pool_size: int = 500,
) -> Tuple[
    float,
    float,
    float,
    Dict[int, float],
    Dict[int, float],
    float,
    Dict[int, float],
    Dict[int, float],
    float,
]:
    """
    Evaluate link prediction with AUC, AP, AUPR, Recall@K, Precision@K, MRR, and Local variants.
    """
    encoder.eval()
    mp_edges = build_message_passing_edges(data, rel)
    z_dict = encoder(data.x_dict, mp_edges)

    src, _, dst = rel
    if split == "val":
        pos_edge_index = data[rel].val_edge_index
    elif split == "test":
        pos_edge_index = data[rel].test_edge_index
    else:
        raise ValueError(f"Unknown split: {split}")
    
    neg_edge_index = negative_sampling(
        pos_edge_index,
        num_nodes=(data[src].num_nodes, data[dst].num_nodes),
        num_neg_samples=num_neg,
        method="sparse",
    )

    pos_scores = predictor(z_dict, pos_edge_index, src, dst).sigmoid()
    neg_scores = predictor(z_dict, neg_edge_index, src, dst).sigmoid()

    y_true = torch.cat(
        [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0
    ).cpu()
    y_scores = torch.cat([pos_scores, neg_scores], dim=0).cpu()

    try:
        from sklearn.metrics import (
            average_precision_score,
            precision_recall_curve,
            roc_auc_score,
            auc as sk_auc,
        )
    
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        aupr = sk_auc(recall, precision)
    except Exception:
        preds = (y_scores >= 0.5).float()
        auc = (preds == y_true).float().mean().item()
        ap = auc
        aupr = auc
    
    recall_at_k = {}
    precision_at_k = {}
    mrr = 0.0
    local_recall_at_k = {}
    local_precision_at_k = {}
    local_mrr = 0.0
    
    if compute_ranking_metrics:
        recall_at_k, precision_at_k, mrr = compute_recall_at_k_and_mrr(
            encoder, predictor, data, rel, pos_edge_index, 
            k_values=k_values, top_k_candidates=top_k_candidates
        )
    
    if compute_local_metrics:
        local_recall_at_k, local_precision_at_k, local_mrr = compute_local_recall_at_k_and_mrr(
            encoder, predictor, data, rel, pos_edge_index,
            k_values=k_values, local_candidate_pool_size=local_candidate_pool_size
        )
    
    return (
        float(auc),
        float(ap),
        float(aupr),
        recall_at_k,
        precision_at_k,
        float(mrr),
        local_recall_at_k,
        local_precision_at_k,
        float(local_mrr),
    )


def print_graph_details(data: HeteroData) -> None:
    """Print  information about the graph."""
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
    
    print()
    
    for edge_type in data.edge_types:
        src, rel_name, dst = edge_type
        if rel_name in ["acted_in", "directed"]:
            edge_store = data[edge_type]
            total_edges = edge_store.edge_index.size(1) if edge_store.edge_index is not None else 0
            
            train_edges = 0
            val_edges = 0
            test_edges = 0
            
            if hasattr(edge_store, "train_edge_index") and edge_store.train_edge_index is not None:
                train_edges = edge_store.train_edge_index.size(1)
            if hasattr(edge_store, "val_edge_index") and edge_store.val_edge_index is not None:
                val_edges = edge_store.val_edge_index.size(1)
            if hasattr(edge_store, "test_edge_index") and edge_store.test_edge_index is not None:
                test_edges = edge_store.test_edge_index.size(1)
            
            print(f"{rel_name} edges ({src} -> {dst}):")
            print(f"  Total: {total_edges}")
            if train_edges > 0:
                print(f"  Training: {train_edges}")
            if val_edges > 0:
                print(f"  Validation: {val_edges}")
            if test_edges > 0:
                print(f"  Test: {test_edges}")
            print()
    
    print("="*80 + "\n")


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    data: HeteroData = torch.load(args.data_path, weights_only=False, map_location=device)
    rel = ("person", args.relation, "movie")
    data = get_edge_split(data, rel, args.train_ratio, args.val_ratio)
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
    predictor = MLPLinkPredictor(
        hidden_channels=args.hidden_channels,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    src, _, dst = rel
    train_edges = data[rel].train_edge_index
    num_train = train_edges.size(1)
    neg_per_pos = max(1, int(args.neg_ratio))
    num_neg_train = num_train * neg_per_pos
    best_val_auc, best_epoch = 0.0, -1

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        predictor.train()
        optimizer.zero_grad()
        mp_edges = build_message_passing_edges(data, rel)
        z_dict = encoder(data.x_dict, mp_edges)

        neg_edge_index = sample_movie_conditioned_negatives(
            train_edges,
            num_src_nodes=data[src].num_nodes,
            neg_per_pos=neg_per_pos,
            device=device,
        )

        pos_scores = predictor(z_dict, train_edges, src, dst)
        neg_scores = predictor(z_dict, neg_edge_index, src, dst)
        
        if args.loss_type == "bpr":
            loss = bpr_loss(pos_scores, neg_scores, sample_negatives=True)
        else:  
            loss = bce_loss(pos_scores, neg_scores)
        
        loss.backward()
        optimizer.step()
        
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            (
                val_auc,
                val_ap,
                val_aupr,
                val_recall_at_k,
                val_precision_at_k,
                val_mrr,
                val_local_recall_at_k,
                val_local_precision_at_k,
                val_local_mrr,
            ) = evaluate(
                encoder,
                predictor,
                data,
                rel,
                num_neg_train,
                split="val",
                compute_ranking_metrics=True,
                k_values=args.k_values,
                top_k_candidates=args.top_k_candidates,
                compute_local_metrics=True,
                local_candidate_pool_size=args.local_candidate_pool_size,
            )
            is_best = val_auc > best_val_auc
            if is_best:
                best_val_auc, best_epoch = val_auc, epoch
                best_state = {
                    "encoder": encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                    "val_aupr": val_aupr,
                    "val_recall_at_k": val_recall_at_k,
                    "val_precision_at_k": val_precision_at_k,
                    "val_mrr": val_mrr,
                    "val_local_recall_at_k": val_local_recall_at_k,
                    "val_local_precision_at_k": val_local_precision_at_k,
                    "val_local_mrr": val_local_mrr,
                }
            # Format recall@k and precision@k strings
            recall_str = " | ".join([f"R@{k} {val_recall_at_k[k]:.4f}" for k in args.k_values])
            precision_str = " | ".join([f"P@{k} {val_precision_at_k[k]:.4f}" for k in args.k_values])
            local_recall_str = " | ".join([f"LR@{k} {val_local_recall_at_k[k]:.4f}" for k in args.k_values])
            local_precision_str = " | ".join([f"LP@{k} {val_local_precision_at_k[k]:.4f}" for k in args.k_values])
            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                f"Val AUC {val_auc:.4f} | Val AP {val_ap:.4f} | Val AUPR {val_aupr:.4f} | "
                f"Val MRR {val_mrr:.4f} | {recall_str} | {precision_str} | "
                f"Local MRR {val_local_mrr:.4f} | {local_recall_str} | {local_precision_str}"
                f"{' <- best' if is_best else ''}"
            )

    # Final test using the best checkpoint.
    if best_epoch != -1:
        encoder.load_state_dict(best_state["encoder"])
        predictor.load_state_dict(best_state["predictor"])
        num_test = data[rel].test_edge_index.size(1)
        num_neg_test = max(1, int(num_test * args.neg_ratio))
        (
            test_auc,
            test_ap,
            test_aupr,
            test_recall_at_k,
            test_precision_at_k,
            test_mrr,
            test_local_recall_at_k,
            test_local_precision_at_k,
            test_local_mrr,
        ) = evaluate(
            encoder,
            predictor,
            data,
            rel,
            num_neg_test,
            split="test",
            compute_ranking_metrics=True,
            k_values=args.k_values,
            top_k_candidates=args.top_k_candidates,
            compute_local_metrics=True,
            local_candidate_pool_size=args.local_candidate_pool_size,
        )
        # Format recall@k and precision@k strings
        test_recall_str = " | ".join([f"R@{k} {test_recall_at_k[k]:.4f}" for k in args.k_values])
        test_precision_str = " | ".join([f"P@{k} {test_precision_at_k[k]:.4f}" for k in args.k_values])
        test_local_recall_str = " | ".join([f"LR@{k} {test_local_recall_at_k[k]:.4f}" for k in args.k_values])
        test_local_precision_str = " | ".join([f"LP@{k} {test_local_precision_at_k[k]:.4f}" for k in args.k_values])
        val_recall_str = " | ".join([f"R@{k} {best_state['val_recall_at_k'][k]:.4f}" for k in args.k_values])
        val_precision_str = " | ".join([f"P@{k} {best_state['val_precision_at_k'][k]:.4f}" for k in args.k_values])
        val_local_recall_str = " | ".join([f"LR@{k} {best_state['val_local_recall_at_k'][k]:.4f}" for k in args.k_values])
        val_local_precision_str = " | ".join([f"LP@{k} {best_state['val_local_precision_at_k'][k]:.4f}" for k in args.k_values])
        
        print("\n" + "="*80)
        print("FINAL TEST RESULTS:")
        print("="*80)
        print(
            f"Best epoch: {best_epoch} | "
            f"Val AUC: {best_state['val_auc']:.4f} | Val AP: {best_state['val_ap']:.4f} | Val AUPR: {best_state['val_aupr']:.4f} | "
            f"Val MRR: {best_state['val_mrr']:.4f} | Val Local MRR: {best_state['val_local_mrr']:.4f}"
        )
        print(f"Val Recall@K: {val_recall_str}")
        print(f"Val Precision@K: {val_precision_str}")
        print(f"Val Local Recall@K: {val_local_recall_str}")
        print(f"Val Local Precision@K: {val_local_precision_str}")
        print(
            f"Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f} | Test AUPR: {test_aupr:.4f} | "
            f"Test MRR: {test_mrr:.4f} | Test Local MRR: {test_local_mrr:.4f}"
        )
        print(f"Test Recall@K: {test_recall_str}")
        print(f"Test Precision@K: {test_precision_str}")
        print(f"Test Local Recall@K: {test_local_recall_str}")
        print(f"Test Local Precision@K: {test_local_precision_str}")
        print("="*80)
    else:
        print("Training finished without validation improvements.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an HGT model for edge prediction (cast recommendation)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="imdb-hetero/data/processed/imdb_hetero_graph_splits.pt"
    )
    parser.add_argument(
        "--relation",
        type=str,
        default="acted_in"
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
        choices=["bce", "bpr"],
        default="bpr"
    )
    parser.add_argument(
        "--neg_ratio",
        type=float,
        default=3.0
    )
    parser.add_argument(
        "--normalize_features",
        action="store_true"
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 50]
    )
    parser.add_argument(
        "--top_k_candidates",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--local_candidate_pool_size",
        type=int,
        default=500
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
