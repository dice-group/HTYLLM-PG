# Distributed Data Processor

Tools for two simple steps: (1) load each sharded language file and tokenize it, (2) merge all tokenized shards into one Hugging Face dataset. Once the flow works for a single language subset + tokenizer, the same scripts repeat it for every subset/tokenizer combination you need.

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

```
/scratch/.../tokenized/hierarchical_adapter/
├── llama-3.1-8B_tokenizer/{5,10,46,95,199}_langs
├── llama-3.2-1B_tokenizer/{...}
└── llama-3.2-3B_tokenizer/{...}
```

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