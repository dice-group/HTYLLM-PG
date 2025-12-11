#!/usr/bin/env python3
"""
Master script to produce clustering variants for FLORES languages.

Outputs:
  processed_artifacts/clusters_llama31.json
  processed_artifacts/clusters_glot500.json
  processed_artifacts/clusters_lang2vec.json
"""

import json
import subprocess
from pathlib import Path

DATA_PREP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = DATA_PREP_DIR / "lang_cluster_analysis"
PROCESSED = DATA_PREP_DIR / "processed_artifacts"


def run(cmd, **kwargs):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def ensure_embeddings(model_key: str):
    output_name = f"flores_embeddings_{model_key}.csv"
    output_path = PROCESSED / output_name
    if output_path.exists():
        print(f"[skip] {output_name} already exists")
        return output_path
    print(f"[gen] creating {output_name}")
    run(
        [
            "python",
            str(DATA_PREP_DIR / "embed_flores_langs.py"),
            "--model-key",
            model_key,
        ]
    )
    return output_path


def cluster_embeddings(model_key: str, auto_k: bool = True):
    csv_path = ensure_embeddings(model_key)
    output_path = PROCESSED / f"clusters_{model_key}.json"
    plots_dir = PROCESSED / "plots" / model_key
    args = [
        "python",
        str(SCRIPT_DIR / "cluster_embeddings.py"),
        "--input",
        str(csv_path),
        "--output",
        str(output_path),
        "--method",
        "kmeans",
        "--scale",
        "--plots-dir",
        str(plots_dir),
    ]
    if auto_k:
        args.append("--auto-k")
    run(args)
    return output_path


def cluster_lang2vec(auto_k: bool = True):
    output_path = PROCESSED / "clusters_lang2vec.json"
    plots_dir = PROCESSED / "plots" / "lang2vec"
    args = [
        "python",
        str(SCRIPT_DIR / "cluster_lang2vec_distances.py"),
        "--distance-type",
        "genetic",
        "--output",
        str(output_path),
        "--method",
        "average",
        "--plots-dir",
        str(plots_dir),
    ]
    if auto_k:
        args.append("--auto-k")
    run(args)
    return output_path


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    llama = cluster_embeddings("llama31_8b")
    glot = cluster_embeddings("glot500")
    lang2vec = cluster_lang2vec()
    summary = {
        "llama31_clusters": str(llama),
        "glot500_clusters": str(glot),
        "lang2vec_clusters": str(lang2vec),
    }
    print("Generated cluster files:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
