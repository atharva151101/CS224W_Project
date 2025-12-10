import json
import argparse
import csv
from pathlib import Path
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm import tqdm

def load_plot_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_text(tokenizer, model, text, device):
    if not text or (isinstance(text, str) and not text.strip()):
        return np.zeros(768, dtype=np.float32)
    
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    
    return embedding.astype(np.float32)

def count_tokens(tokenizer, text):
    if not text or (isinstance(text, str) and not text.strip()):
        return 0
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return len(tokens)

def encode_keywords(tokenizer, model, keywords, device):
    if not keywords or len(keywords) == 0:
        return np.zeros(768, dtype=np.float32)
    
    # Join keywords with commas
    keywords_text = ", ".join(keywords)
    return encode_text(tokenizer, model, keywords_text, device)

def main():
    parser = argparse.ArgumentParser(description='Generate BERT encodings for movie plot data')
    parser.add_argument(
        '--input',
        type=str,
        default='imdb-hetero/data/raw/plot_data.json'
    )
    parser.add_argument(
        '--movies_csv',
        type=str,
        default='imdb-hetero/data/raw/movies.csv'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='imdb-hetero/data/raw/plot_data_encoded.json'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='bert-base-uncased'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None
    )
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading BERT model: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading plot data from: {input_path}")
    plot_data = load_plot_data(input_path)
    print(f"Loaded {len(plot_data)} movies")

    movies_path = Path(args.movies_csv)
    if not movies_path.exists():
        raise FileNotFoundError(f"movies.csv not found at: {movies_path}")
    with movies_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        movie_ids = {
            str(row.get("tconst", "")).strip()
            for row in reader
            if row.get("tconst")
        }
    print(f"Loaded {len(movie_ids)} movie ids from movies.csv")

    filtered_plot_data = []
    for movie in plot_data:
        tid = str(movie.get('title_id') or movie.get('tconst') or "").strip()
        if tid and tid in movie_ids:
            filtered_plot_data.append(movie)
    print(f"Filtered plot data: {len(filtered_plot_data)} entries match movies.csv")
    plot_data = filtered_plot_data
    
    if args.limit:
        plot_data = plot_data[:args.limit]
        print(f"Limited to first {len(plot_data)} movies")
    
    encoded_data = []
    for movie in tqdm(plot_data, desc="Encoding movies"):
        title_id = movie.get('title_id', '')
        title = movie.get('title', '')
        synopsis = movie.get('synopsis', '')
        storyline = movie.get('storyline', '')
        keywords = movie.get('keywords', [])
        
        title_encoding = encode_text(tokenizer, model, title, device)
        
        synopsis_token_count = count_tokens(tokenizer, synopsis)
        storyline_token_count = count_tokens(tokenizer, storyline)
        
        synopsis_text = synopsis
        synopsis_token_count_used = synopsis_token_count
        synopsis_source = 'synopsis'
        token_limit = False
        
        if synopsis_token_count > 512:
            if storyline_token_count > 512:
                token_limit = True
                if synopsis_token_count <= storyline_token_count:
                    synopsis_text = synopsis
                    synopsis_token_count_used = synopsis_token_count
                    synopsis_source = 'synopsis'
                else:
                    synopsis_text = storyline
                    synopsis_token_count_used = storyline_token_count
                    synopsis_source = 'storyline'
            else:
                synopsis_text = storyline
                synopsis_token_count_used = storyline_token_count
                synopsis_source = 'storyline'
        
        synopsis_encoding = encode_text(tokenizer, model, synopsis_text, device)
        keywords_encoding = encode_keywords(tokenizer, model, keywords, device)
        
        encoded_entry = {
            'tconst': title_id,
            'title': title,
            'title_encoding': title_encoding.tolist(),  
            'synopsis': synopsis,
            'synopsis_encoding': synopsis_encoding.tolist(),
            'synopsis_token_count': synopsis_token_count_used,
            'synopsis_source': synopsis_source,  
            'keywords': keywords,
            'keywords_encoding': keywords_encoding.tolist(),
            'token_limit': token_limit
        }
              
        encoded_data.append(encoded_entry)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving encoded data to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(encoded_data, f, indent=2, ensure_ascii=False)
    
    print(f"Encoded {len(encoded_data)} movies")
    print(f"Each encoding has dimension: {len(title_encoding)}")

if __name__ == '__main__':
    main()

