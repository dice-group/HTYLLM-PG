from pathlib import Path
from datasets import load_from_disk

VAL_FRACTION = 0.02
SEED = 42
ROOTS = [
    Path("/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/llama-3.1-8B_tokenizer"),
]
SUBSETS = ["5_langs", "10_langs", "46_langs"]


def split_dataset(src: Path, dst: Path):
    if not src.exists():
        print(f"[WARN] missing {src}, skipping")
        return
    if dst.exists():
        print(f"[SKIP] {dst} already exists")
        return

    ds = load_from_disk(str(src))
    if hasattr(ds, "keys") and "validation" in ds:
        print(f"[INFO] {src} already split; copying to {dst}")
        ds.save_to_disk(str(dst))
        return

    split = ds.train_test_split(test_size=VAL_FRACTION, seed=SEED)
    split.save_to_disk(str(dst))
    print(f"[DONE] saved split to {dst}")


def main():
    for root in ROOTS:
        for subset in SUBSETS:
            src = root / subset
            dst = root / f"{subset}_split"
            split_dataset(src, dst)


if __name__ == "__main__":
    main()
