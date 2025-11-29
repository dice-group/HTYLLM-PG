import argparse
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
DEFAULT_BATCH_SIZE = int(os.environ.get("FLORES_BATCH_SIZE", 48))

MODEL_CONFIGS = {
    "llama31_8b": {
        "model_name": os.environ.get("FLORES_LLAMA_MODEL", "meta-llama/Llama-3.1-8B"),
        "output": DATA_DIR / os.environ.get("FLORES_LLAMA_OUTPUT", "flores_embeddings_llama31_8b.csv"),
    },
    "glot500": {
        "model_name": os.environ.get("FLORES_GLOT_MODEL", "cis-lmu/Glot500"),
        "output": DATA_DIR / os.environ.get("FLORES_GLOT_OUTPUT", "flores_embeddings_glot500.csv"),
    },
}


def load_model(name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    return tok, model, device


def encode_batch(texts: List[str], tok, model, device):
    inp = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        h = model(**inp).last_hidden_state
    mask = inp["attention_mask"].unsqueeze(-1).expand_as(h).float()
    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()


def embed_language(ds, tok, model, device, bs: int) -> Optional[np.ndarray]:
    texts = [row["text"] for row in ds]
    vecs = []

    for i in range(0, len(texts), bs):
        try:
            vecs.append(encode_batch(texts[i:i+bs], tok, model, device))
        except Exception:
            pass

    if not vecs:
        return None
    return np.mean(np.vstack(vecs), axis=0)


def parse_args():
    p = argparse.ArgumentParser(description="Embed FLORES languages.")
    p.add_argument("--model-key", choices=["all", *MODEL_CONFIGS.keys()], default="all", help="Model to run.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    return p.parse_args()


def run_embedding_job(key: str, languages, meta, datasets, bs: int):
    cfg = MODEL_CONFIGS[key]
    tok, model, device = load_model(cfg["model_name"])

    rows, embs, missing = [], [], []

    for subset in tqdm(languages, desc=key):
        ds = datasets.get(subset)
        if ds is None:
            missing.append(subset)
            continue
        vec = embed_language(ds, tok, model, device, bs)
        if vec is None:
            missing.append(subset)
            continue
        rows.append({"subset": subset})
        embs.append(vec)

    if not embs:
        raise RuntimeError(f"No embeddings for {cfg['model_name']}")

    dim = model.config.hidden_size
    emb_df = pd.DataFrame(embs, columns=[f"llm_emb_{i}" for i in range(dim)])
    out_df = pd.merge(pd.DataFrame(rows), meta, on="subset", how="left")
    pd.concat([out_df, emb_df], axis=1).to_csv(cfg["output"], index=False)

    print(f"Saved {cfg['model_name']} to {cfg['output']}")
    if missing:
        print(f"Missing {len(missing)} languages")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    if not LANG_PATH.exists():
        sys.exit(f"Missing language file: {LANG_PATH}")

    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(LANG_PATH)
    languages = meta["subset"].tolist()

    datasets = {}
    for subset in languages:
        try:
            datasets[subset] = load_dataset(DATASET_NAME, subset, split=SPLIT)
        except Exception:
            datasets[subset] = None

    keys = MODEL_CONFIGS.keys() if args.model_key == "all" else [args.model_key]
    for key in keys:
        run_embedding_job(key, languages, meta, datasets, args.batch_size)


if __name__ == "__main__":
    main()
