#!/usr/bin/env python
"""
rank_sentence_scores_one_rank.py
--------------------------------
Compute the joint sentence score (Rj) for a **single** rank_* folder.
The algorithm follows the paper (Algorithm 2) – local popularity
(Rl), global importance (Rg via PageRank), and their weighted
combination (Rj = α·Rl + β·Rg) [1].

Usage
-----
    python rank_sentence_scores_one_rank.py \
        --rank-dir  /path/to/rank_00000 \
        --global-counts-dir /path/to/global_counts \
        --tokenizer meta-llama/Llama-3.1-8B \
        --alpha 0.5 --beta 0.5
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import tempfile
import numpy as np
import scipy.sparse as sp
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ----------------------------------------------------------------------
# Helper: load the pre‑computed word / sub‑word counts (produced by the
# counting step you ran earlier).  They are stored in
# global_counts/word_counts.json and subword_counts.json.
# ----------------------------------------------------------------------
def load_global_counts(counts_dir: Path):
    wc_path = counts_dir / "word_counts.json"
    swc_path = counts_dir / "subword_counts.json"
    if not wc_path.is_file() or not swc_path.is_file():
        raise FileNotFoundError(f"Missing count files in {counts_dir}")
    with wc_path.open() as f:
        word_counts = json.load(f)
    with swc_path.open() as f:
        subword_counts = json.load(f)
    return word_counts, subword_counts


# ----------------------------------------------------------------------
# 1️⃣  Build the word‑co‑occurrence matrix (window = 5) for this rank
# ----------------------------------------------------------------------
def build_cooc(rank_dir: Path, tokenizer, window: int = 5):
    vocab: dict[str, int] = {}
    rows, cols = [], []

    ds: Dataset = load_from_disk(str(rank_dir))
    for txt in tqdm(ds["text"], desc=f"Co‑occurrence in {rank_dir.name}"):
        words = txt.split()
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)
        for i, wi in enumerate(words):
            idx_i = vocab[wi]
            # forward window
            for j in range(i + 1, min(i + window + 1, len(words))):
                wj = words[j]
                idx_j = vocab[wj]
                rows.append(idx_i); cols.append(idx_j)
                rows.append(idx_j); cols.append(idx_i)   # symmetry
    data = np.ones(len(rows), dtype=np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(len(vocab), len(vocab)))
    X.setdiag(0)
    X.eliminate_zeros()
    # ordered list of words for later lookup
    ordered_vocab = [None] * len(vocab)
    for w, i in vocab.items():
        ordered_vocab[i] = w
    return X, ordered_vocab


# ----------------------------------------------------------------------
# 2️⃣  PageRank on the co‑occurrence graph
# ----------------------------------------------------------------------
def pagerank(X, damping=0.85, max_iter=100, tol=1e-6):
    n = X.shape[0]
    out = np.array(X.sum(axis=1)).flatten()
    out[out == 0] = 1.0                     # avoid division by zero
    D_inv = sp.diags(1.0 / out)
    M = D_inv @ X
    pr = np.full(n, 1.0 / n)
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        pr_new = damping * (M.T @ pr) + teleport
        if np.linalg.norm(pr_new - pr, 1) < tol:
            break
        pr = pr_new
    return pr


# ----------------------------------------------------------------------
# 3️⃣  Local popularity (Rl) – uses the global word counts
# ----------------------------------------------------------------------
def compute_local_scores(word_counts: dict, subword_counts: dict, tokenizer):
    Rl = {}
    for w, wc in word_counts.items():
        sub = tokenizer.tokenize(w)
        Rl[w] = sum(subword_counts[t] for t in sub) - wc
    return Rl


# ----------------------------------------------------------------------
# 4️⃣  Add the joint_score column to the rank dataset
# ----------------------------------------------------------------------
import shutil
import tempfile
from pathlib import Path

def add_joint_score(rank_dir: Path, tokenizer, alpha, beta,
                   Rl: dict, Rg: dict):
    """
    Load the rank, compute joint_score for every sentence,
    and write the result to a *new* directory.  Afterwards
    replace the original rank folder with the new one.
    """
    ds: Dataset = load_from_disk(str(rank_dir))

    # ---------- compute the column ----------
    def score_batch(batch):
        scores = []
        for txt in batch["text"]:
            s = 0.0
            for w in txt.split():
                s += alpha * Rl.get(w, 0.0) + beta * Rg.get(w, 0.0)
            scores.append(s)
        return {"joint_score": scores}

    ds = ds.map(score_batch, batched=True, batch_size=1000, remove_columns=None)

    # ---------- write to a temporary location ----------
    with tempfile.TemporaryDirectory(prefix="rank_scored_") as tmp_dir:
        tmp_path = Path(tmp_dir) / rank_dir.name   # e.g. tmp/.../rank_00000
        ds.save_to_disk(str(tmp_path))

        # ---------- atomically replace the original folder ----------
        # 1) move the original folder out of the way (so we keep a backup)
        backup_path = rank_dir.parent / f"{rank_dir.name}_old"
        if backup_path.exists():
            shutil.rmtree(str(backup_path))
        rank_dir.rename(backup_path)

        # 2) move the new scored folder into place
        shutil.move(str(tmp_path), str(rank_dir))

        # 3) optionally delete the backup (remove if you’re sure)
        shutil.rmtree(str(backup_path))

    print(f"✅ {rank_dir.name} – joint_score added and folder replaced")


# ----------------------------------------------------------------------
# Main driver (single rank)
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute joint_score for ONE rank_* folder "
                    "(distributed version of rank_sentence_scores).")
    parser.add_argument("--rank-dir", type=Path, required=True,
                        help="Path to a single rank_XXXXX directory.")
    parser.add_argument("--global-counts-dir", type=Path, required=True,
                        help="Directory that holds word_counts.json & subword_counts.json.")
    parser.add_argument("--tokenizer", type=str,
                        default="meta-llama/Llama-3.1-8B",
                        help="HF tokenizer name.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for local score (Rl).")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Weight for global score (Rg).")
    parser.add_argument("--window", type=int, default=5,
                        help="Context window size for co‑occurrence (paper uses 5).")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    # ----- load pre‑computed WC / SWC  (Algorithm 2 line 25‑26) -----
    word_counts, subword_counts = load_global_counts(args.global_counts_dir)

    # ----- build co‑occurrence matrix for THIS rank -----
    X, vocab = build_cooc(args.rank_dir, tokenizer, window=args.window)

    # ----- PageRank → global importance Rg (Algorithm 2 line 26) -----
    pr = pagerank(X)                                 # [1]
    Rg = { vocab[i]: float(pr[i]) for i in range(len(vocab)) }

    # ----- local popularity Rl (Algorithm 2 line 25) -----
    Rl = compute_local_scores(word_counts, subword_counts, tokenizer)

    # ----- write joint_score back to the rank dataset -----
    add_joint_score(args.rank_dir, tokenizer, args.alpha, args.beta, Rl, Rg)


if __name__ == "__main__":
    main()