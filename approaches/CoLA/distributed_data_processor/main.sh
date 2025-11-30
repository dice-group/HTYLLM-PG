#!/usr/bin/env bash
set -euo pipefail


# Legacy run (tokenize + merge only)
# bash "tokenize_all_tokenizers_lang_combinations.sh" \
#   --shard-dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples \
#   --output-base /scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter \
#   --job-prefix full_tok

echo "[INFO] Launching pipeline with ranking INLCUDING per-language top-K selection enabled. change main.sh if you dont want to sue it"
bash "tokenize_all_tokenizers_lang_combinations.sh" \
  --shard-dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples \
  --output-base /scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter \
  --job-prefix ranked_tok \
  --enable-ranking \
  --topk-sizes "10000 20000 30000" \
  --topk-min-language-size 30000
