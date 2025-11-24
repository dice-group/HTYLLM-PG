import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "processed_artifacts")
LANG_PATH = os.path.join(DATA_DIR, "filtered_languages.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "flores_embeddings.csv")
DATASET_NAME = "openlanguagedata/flores_plus"
SPLIT = "dev"

SAMPLES_PER_LANG = int(os.environ.get("FLORES_SAMPLE_COUNT", 50)) 
BATCH_SIZE = 16
MODEL_NAME = os.environ.get("FLORES_MODEL_NAME", "meta-llama/Llama-3.1-8B")

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device).eval()
    
    return tokenizer, model, device, dtype

def encode_batch(texts: List[str], tokenizer, model, device):
    """
    Encodes a batch of sentences using MASKED MEAN POOLING.
    Superior for clustering tasks compared to last-token pooling.
    """
    # 1. Tokenize as a batch (padding is now CRITICAL)
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True,       # Pad shortest to match longest in this batch
        truncation=True, 
        max_length=512
    ).to(device)

    # 2. Inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get all hidden states: Shape [Batch_Size, Seq_Len, Hidden_Dim]
    token_embeddings = outputs.last_hidden_state

    # 3. Masked Mean Pooling Logic
    # We want to average the vectors, but NOT the padding vectors (which are zeros/garbage).
    
    # Create a mask that matches the shape of the embeddings
    # attention_mask is [Batch, Seq], we need [Batch, Seq, Dim]
    input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # sum the embeddings of real tokens (multiply by 0 if its padding)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    
    # Sum the number of real tokens (to divide by)
    # clamp(min=1e-9) prevents "Divide by Zero" errors if a sentence happens to be empty
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Calculate Mean
    mean_embeddings = sum_embeddings / sum_mask
    mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
    
    return mean_embeddings.float().cpu().numpy()

def fetch_texts(subset: str) -> List[str]:
    # Only fetch exactly what we need to save RAM/Time
    ds = load_dataset(DATASET_NAME, subset, split=f"{SPLIT}[:{SAMPLES_PER_LANG}]")
    return [row["text"] for row in ds]

def main():
    if not os.path.exists(LANG_PATH):
        print(f"Language metadata missing at {LANG_PATH}.")
        sys.exit(1)

    languages = pd.read_csv(LANG_PATH)["subset"].tolist()
    tokenizer, model, device, dtype = load_model()
    dim = model.config.hidden_size

    rows = []
    embeddings = []
    missing = []

    print(f"Processing {len(languages)} languages with Batch Size {BATCH_SIZE}...")
    
    for subset in tqdm(languages, desc="Languages"):
        try:
            texts = fetch_texts(subset)
        except Exception as exc:
            # Handle dataset loading errors (e.g., wrong codes)
            missing.append(subset)
            continue

        if not texts:
            missing.append(subset)
            continue

        # --- BATCH PROCESSING LOOP ---
        lang_vectors = []
        
        # Process in chunks of BATCH_SIZE
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            try:
                batch_emb = encode_batch(batch_texts, tokenizer, model, device)
                lang_vectors.append(batch_emb)
            except Exception as exc:
                print(f"[warn] Batch failed for {subset}: {exc}")
                continue

        if lang_vectors:
            # Concatenate all batches for this language
            all_vecs = np.vstack(lang_vectors)
            # Average them to get the single "Language Vector"
            avg_vec = np.mean(all_vecs, axis=0)
            
            embeddings.append(avg_vec)
            rows.append({"subset": subset})
        else:
            missing.append(subset)

    # Save Results
    if not embeddings:
        raise RuntimeError("No embeddings produced.")

    emb_df = pd.DataFrame(embeddings, columns=[f"llm_emb_{i}" for i in range(dim)])
    meta_df = pd.DataFrame(rows)
    
    # Merge with original metadata to keep script/region info
    result = pd.merge(meta_df, pd.read_csv(LANG_PATH), on="subset", how="left")
    final_df = pd.concat([result, emb_df], axis=1)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\nSuccess! Saved to {OUTPUT_PATH}")
    if missing:
        print(f"Skipped languages: {missing[:5]}... (Total: {len(missing)})")

if __name__ == "__main__":
    main()