import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent / "processed_artifacts"
LANG_PATH = DATA_DIR / "filtered_languages.csv"
DATASET_NAME = "openlanguagedata/flores_plus"
SPLIT = "dev"
BATCH_SIZE = 16

MODEL_CONFIGS = [
    {
        "name": os.environ.get("FLORES_LLAMA_MODEL", "meta-llama/Llama-3.1-8B"),
        "output": DATA_DIR / os.environ.get("FLORES_LLAMA_OUTPUT", "flores_embeddings_llama31_8b.csv"),
    },
    {
        "name": os.environ.get("FLORES_GLOT_MODEL", "cis-lmu/Glot500"),
        "output": DATA_DIR / os.environ.get("FLORES_GLOT_OUTPUT", "flores_embeddings_glot500.csv"),
    },
]


def load_model(name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    return tokenizer, model.to(device).eval(), device


def encode_batch(texts: List[str], tokenizer, model, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        h = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).expand_as(h).float()
    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()


def embed_language(subset: str, tokenizer, model, device) -> Optional[np.ndarray]:
    try:
        ds = load_dataset(DATASET_NAME, subset, split=SPLIT)
    except Exception:
        return None

    vecs = []
    texts = [row["text"] for row in ds]

    for i in range(0, len(texts), BATCH_SIZE):
        try:
            vecs.append(encode_batch(texts[i:i+BATCH_SIZE], tokenizer, model, device))
        except Exception:
            pass

    if not vecs:
        return None
    return np.mean(np.vstack(vecs), axis=0)


def main():
    if not LANG_PATH.exists():
        sys.exit(f"Missing language file: {LANG_PATH}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    language_meta = pd.read_csv(LANG_PATH)
    languages = language_meta["subset"].tolist()

    for cfg in MODEL_CONFIGS:
        name, out_file = cfg["name"], cfg["output"]
        print(f"\nEmbedding FLORES {SPLIT} with {name}...")

        tokenizer, model, device = load_model(name)
        rows, embs, missing = [], [], []

        for subset in tqdm(languages, desc=name):
            vec = embed_language(subset, tokenizer, model, device)
            if vec is None:
                missing.append(subset)
                continue
            embs.append(vec)
            rows.append({"subset": subset})

        if not embs:
            raise RuntimeError(f"No embeddings for {name}")

        emb_df = pd.DataFrame(embs, columns=[f"llm_emb_{i}" for i in range(model.config.hidden_size)])
        result = pd.merge(pd.DataFrame(rows), language_meta, on="subset", how="left")
        pd.concat([result, emb_df], axis=1).to_csv(out_file, index=False)

        print(f"Saved to {out_file}")
        if missing:
            print(f"Missing: {missing[:5]} (total {len(missing)})")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
