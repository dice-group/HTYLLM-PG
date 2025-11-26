#!/usr/bin/env bash
set -euo pipefail

COUNT_ROOT=$1
FINAL_OUT="${COUNT_ROOT}/global_counts"
mkdir -p "${FINAL_OUT}"

python - <<PY
import json, pathlib, sys
from collections import Counter

root = pathlib.Path("$COUNT_ROOT")
global_word = Counter()
global_sub = Counter()

for rank_dir in sorted(root.iterdir()):
    if not rank_dir.is_dir(): continue
    wc_path = rank_dir / "word_counts.json"
    swc_path = rank_dir / "subword_counts.json"
    if wc_path.is_file():
        global_word.update(json.load(wc_path.open()))
    if swc_path.is_file():
        global_sub.update(json.load(swc_path.open()))

pathlib.Path("$FINAL_OUT").mkdir(parents=True, exist_ok=True)
(pathlib.Path("$FINAL_OUT") / "word_counts.json").write_text(json.dumps(global_word, indent=2))
(pathlib.Path("$FINAL_OUT") / "subword_counts.json").write_text(json.dumps(global_sub, indent=2))
print("✅ Global counts written to", "$FINAL_OUT")
PY