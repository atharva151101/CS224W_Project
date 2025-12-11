# Execution Guide

## Prerequisites

- Python packages: `torch`, `torch-geometric`, `pandas`, `numpy`, `scikit-learn`, `transformers`, `scipy`
- Raw data in `imdb-hetero/data/raw/`: `movies.csv`, `people.csv`, `actor_director_movies_edges.csv`, `plot_data.json` (optional)

## Execution Order

### 1. Pre-process and Filter Data

Run the collab notebook pre_process.ipynb


### 2. Encode Plot Data (Optional)
```bash
python plot_encoding.py
```
**Output:** `imdb-hetero/data/raw/plot_data_encoded.json`

### 3. Preprocess Raw Data
```bash
python preprocess_to_artifacts.py
```
**Output:** Feature files in `imdb-hetero/data/processed/` 

### 4. Run Baseline Models (Optional)

Run the collab notebook baseline_models.py

### 3. Build PyG Graph
```bash
python build_pyg_graph.py
```
**Output:** `imdb-hetero/data/processed/imdb_hetero_graph.pt`

### 4. Create Train/Val/Test Splits
```bash
python check_graph_and_make_splits.py
```
**Optional:** Add Laplacian eigenvectors
```bash
python check_graph_and_make_splits.py --add_eigenvectors --num_eigenvectors 128
```
**Output:** `imdb-hetero/data/processed/imdb_hetero_graph_splits.pt`

### 5. Add Graph Statistics (Optional)
```bash
python add_graph_statistics.py
```
**Output:** `imdb-hetero/data/processed/imdb_hetero_graph_with_stats.pt`

### 6. Train Model

**Option A: Rating Prediction (Node Regression)**
```bash
python train_rating_predictor.py
```
**If using graph statistics (Step 5):**
```bash
python train_rating_predictor.py --data_path imdb-hetero/data/processed/imdb_hetero_graph_with_stats.pt
```
**Output:** `best_rating_predictor.pt`, `predictions.csv`

**Option B: Link Prediction (Cast Recommendation)**
```bash
python HGT.py 
```
**If using graph statistics (Step 5):**
```bash
python HGT.py --data_path imdb-hetero/data/processed/imdb_hetero_graph_with_stats.pt 
```
**Output:** Model checkpoint and link prediction metrics



## Complete Pipeline (With All Features)

```bash
python plot_encoding.py
python preprocess_to_artifacts.py --plot_encoded_json imdb-hetero/data/raw/plot_data_encoded.json
python build_pyg_graph.py
python check_graph_and_make_splits.py --add_eigenvectors --num_eigenvectors 128
python add_graph_statistics.py
python train_rating_predictor.py --num_layers 3 --normalize_features --epochs 1000 --dropout 0.3
python HGT_colab.py --hidden_channels 256 --num_layers 4 --normalize_features --epochs 50 --dropout 0.3 --top_k_candidates 2000 --loss_type bpr
```
