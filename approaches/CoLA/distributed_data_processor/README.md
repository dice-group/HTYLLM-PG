# Distributed Data Processor

Tools for two simple steps: (1) load each sharded language file and tokenize it, (2) merge all tokenized shards into one Hugging Face dataset. Once the flow works for a single language subset + tokenizer, the same scripts repeat it for every subset/tokenizer combination you need.

```
/scratch/.../tokenized/hierarchical_adapter/
├── llama-3.1-8B_tokenizer/{5,10,46,95,199}_langs
├── llama-3.2-1B_tokenizer/{...}
└── llama-3.2-3B_tokenizer/{...}
```

Each tokenized example keeps a `language` column (derived from the source shard) and a deterministic `split` label. During merging we build a `DatasetDict(train, validation)` where the validation slice contains roughly 5 % of every language (configurable via `merge_tokenized_ranks.py --split_fraction`). Point any training job at the merged path and both splits are already present.

```
Sharded JSONL(.gz) per language
          │
          ▼
tokenize_and_merge_pipeline.sh
          │
          ├── Tokenization array → <output>_ranks/rank_*
          ▼
  merge_tokenized_ranks.py
          │
          ▼
 Final HF dataset @ <output>/ (DatasetDict with train + validation)
```

## What you need

- SLURM with `sbatch`.
- Shards laid out like `/scratch/.../sharded_samples/<lang>/*.jsonl.gz`.
- Tokenizers available locally (HF cache or on disk).

## Where the data lives

```
/scratch/hpc-prf-merlin/project_data/moe_study/
├── fw_samples/
│   └── sharded_samples/
│       ├── eng_Latn/00000.jsonl_part_00000.jsonl.gz
│       ├── rus_Cyrl/...
│       └── ...
└── tokenized/
    └── hierarchical_adapter/
        └── <tokenizer_name>/<subset>/
```

1. Tokenization jobs read the shards under `fw_samples/sharded_samples/<lang>/...`.
2. Each SLURM rank writes its tokenized Hugging Face dataset to `<output-root>_ranks/rank_<id>/`.
3. The merge job concatenates those rank datasets and saves the final dataset to `<output-root>/`.

Example for `--output-root /scratch/.../tokenized/hierarchical_adapter/llama-3.2-1B_tokenizer/5_langs`:

```
/scratch/.../llama-3.2-1B_tokenizer/5_langs_ranks/
├── rank_00000/
├── rank_00001/
└── ...
/scratch/.../llama-3.2-1B_tokenizer/5_langs/
├── data-00000-of-00001.arrow
├── dataset_info.json
└── state.json
```

## Quick smoke test

Run a single tokenizer + subset via:

```bash
bash tokenize_and_merge_pipeline.sh \
  --shard-dir /scratch/.../fw_samples/sharded_samples \
  --tokenizer meta-llama/Llama-3.2-1B \
  --language-subset five_representatives_mediods \
  --num-proc 4 \
  --merge-cpus 6 \
  --merge-mem 96G \
  --merge-time 03:00:00 \
  --merge-workers 4 \
  --output-root /scratch/.../tokenized/hierarchical_adapter/llama-3.2-1B_tokenizer/5_langs_smoke \
  --log-root logs/smoke_llama32b \
  --job-prefix smoke_llama32b
```

This submits the tokenization array and its merge job. Check `logs/smoke_llama32b/` for progress.

## Full run

`bash main.sh` launches every tokenizer/subset combo and writes to:

See the `language_subsets.py` lists for the exact language mixes per subset.

### Lang2Vec-driven subset

`language_subsets.py` now exposes `lang2vec_auto_best_languages`, a 12-language mix built from the Lang2Vec clustering pipeline (`data_prep/lang_cluster_analysis/run_lang2vec_best_clusters.sh`). The loader automatically prefers the fresh artifact at `data_prep/processed_artifacts/clusters_lang2vec_genetic_auto_best4x3.json` and falls back to the checked-in preset under `data_prep/lang_cluster_analysis/presets/`. To swap in a new cluster selection, rerun the Lang2Vec script and point the pipeline at the resulting JSON via `export LANG2VEC_CLUSTER_SELECTION=/path/to/your_best4x3.json` before invoking `tokenize_and_merge_pipeline.sh --language-subset lang2vec_auto_best_languages`.

## Resources

- Tokenization ranks always use 4 CPU / 32 GB / 1 h on `normal`.
- Merge jobs auto-scale per subset:

| Subset                      | Ranks | Merge Mem | Merge Time |
|-----------------------------|-------|-----------|------------|
| five_representatives...     | 5     | 64 GB     | 02:00      |
| ten_representatives...      | 10    | 80 GB     | 03:00      |
| twenty_two_representatives… | 22    | 96 GB     | 04:00      |
| fourty_six_representatives… | 46    | 128 GB    | 06:00      |
| ninty_five_representatives… | 95    | 160 GB    | 08:00      |
| hundred_ninty_nine_representatives… | 100 | 192 GB | 10:00 |

Override with `--merge-mem`, `--merge-time`, or `--merge-cpus` if needed.

## Logs & outputs

- Logs land in `logs/<job-prefix>/tokenize_%A_%a.log` and `logs/<job-prefix>/merge_%j.log`.
- Rank outputs go to `<output-root>_ranks/`.
- Final merged dataset saves to `<output-root>/`.

## Verify outputs

Edit the constants at the top of `verify_tokenized_outputs.py` (output base path, report path, etc.), then run:

```bash
python verify_tokenized_outputs.py
```

The script scans every `<tokenizer>/<subset>` dataset, gathers sample/token counts, and writes a Markdown summary (default `logs/dataset_report.md`) with collapsible sections per tokenizer.

## Scripts

- `tokenize_and_merge_pipeline.sh` — runs one tokenizer/subset job.
- `tokenize_all_tokenizers.sh` — loops over all combos.
- `merge_tokenized_ranks.py` — merges `rank_*` datasets with progress logs.
- `main.sh` — runs the full sweep.

---

---
### Helper, trash, note section
original data sample
/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples/

/scratch/hpc-prf-merlin/joel
/scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/distributed_data_processor


hundred_ninty_nine_representatives_mediods
ninty_five_representatives_mediods
fourty_six_representatives_mediods
ten_representatives_mediods
five_representatives_mediods

meta-llama/Llama-3.1-8B
meta-llama/Llama-3.2-3B
meta-llama/Llama-3.2-1B

#### Cleanup helper

for clean up everythign if necessary: 

```bash
BASE=/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter
TOKENIZERS=(llama-3.1-8B_tokenizer llama-3.2-1B_tokenizer llama-3.2-3B_tokenizer)
SUBSETS=(
  5_langs
  10_langs
  22_langs
  46_langs
  95_langs
  199_langs
  eng_plus_5_langs
  eng_plus_10_langs
  eng_plus_22_langs
  eng_plus_46_langs
  eng_plus_95_langs
  eng_plus_199_langs
)
for tok in "${TOKENIZERS[@]}"; do
  for subset in "${SUBSETS[@]}"; do
    TARGET="${BASE}/${tok}/${subset}"
    echo "Resetting ${TARGET}"
    rm -rf "${TARGET}" "${TARGET}_ranks"
    mkdir -p "${TARGET}"
  done
done
```
verify 
```bash
tree /scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter
```
```
rm -rf logs/full_tok_llama-3.*
```

```
ls /scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/llama-3.1-8B_tokenizer
ls /scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/llama-3.2-1B_tokenizer
ls /scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/llama-3.2-3B_tokenizer
```

Only remove rank folders when you no longer need to re-run merges:

```bash
BASE=/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter
TOKENIZERS=(llama-3.1-8B_tokenizer llama-3.2-1B_tokenizer llama-3.2-3B_tokenizer)
SUBSETS=(
  5_langs
  10_langs
  22_langs
  46_langs
  95_langs
  199_langs
  eng_plus_5_langs
  eng_plus_10_langs
  eng_plus_22_langs
  eng_plus_46_langs
  eng_plus_95_langs
  eng_plus_199_langs
)
for tok in "${TOKENIZERS[@]}"; do
  for subset in "${SUBSETS[@]}"; do
    TARGET="${BASE}/${tok}/${subset}_ranks" # we only remove ranks and keep combined datasets
    if [[ -d "${TARGET}" ]]; then
      echo "Removing ${TARGET}"
      rm -rf "${TARGET}"
    fi
  done
done
```

Rank directories persist by design so you can re-run `merge_tokenized_ranks.py` without re-tokenizing data. Delete them to save space only after the merged datasets are verified
