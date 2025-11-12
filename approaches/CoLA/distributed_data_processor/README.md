## Distributed Tokenization Pipeline

purpose. tokenization in three phases so workloads stay balanced across large multilingual corpora

### 1. Shard the raw corpus

```
sbatch shard.sh
```

`shard_corpus.py` walks every language sub-directory, streams the underlying `.jsonl` / `.jsonl.gz` files, and writes uniformly sized shards (`shard_000000.jsonl.gz`, …) plus `shard_manifest.json` with sample counts. Adjust `SOURCE_DIR`, `SHARD_DIR`, and `TARGET_SHARD_BYTES` inside `shard.sh`

### 2. Tokenize shards via SLURM array

```
sbatch tokenize.sh
```

Update `SHARD_DIR`, `TOKENIZED_OUTPUT`, `TOKENIZER_NAME`, and `NUM_PROC` in `tokenize.sh`. 
This launches an slurm array, each task loads its slice of the shard list and writes to `TOKENIZED_OUTPUT/rank_XXXXX`. Because shards are equal in size, runtimes per rank shouldnt vary that much

### 3. Combine & verify

```
sbatch combine.sh
python verify_tokenization.py \
  --manifest /path/to/shard_manifest.json \
  --dataset_dir /path/to/tokenized_fw_combined
```

`combine_tokenized.py` loads every `rank_*/` dataset, concatenates them, and saves a single `load_from_disk` artifact. It also checks counts against the sharding manifest. Run `verify_tokenization.py` on either the combined dataset or the per-rank directory to double-check no samples were lost.

Adjust the paths in `combine.sh` and the verification command for your environment. All scripts default to UTF-8 streaming with minimal memory overhead, so they can run on standard CPU partitions while the tokenizer itself uses Hugging Face datasets for batching.

---
## Data layout on Merlin

Tokenized datasets are at Otus cluster here: `/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hierarchical_adapter/`. 
Each tokenizer gets its own directory, and each tokenizer directory contains the same language-pack buckets. Reference layout:

```
/scratch/.../hierarchical_adapter
├── llama-3.1-8B_tokenizer
│   ├── 5_langs
│   ├── 10_langs
│   ├── 46_langs
│   ├── 95_langs
│   └── 199_langs
├── llama-3.2-1B_tokenizer
│   ├── 5_langs
│   ├── 10_langs
│   ├── 46_langs
│   ├── 95_langs
│   └── 199_langs
└── llama-3.2-3B_tokenizer
    ├── 5_langs
    ├── 10_langs
    ├── 46_langs
    ├── 95_langs
    └── 199_langs
```

Use these directories when pointing scripts at shards, copying tokenized ranks, or staging combined datasets for training.
Use for ablations also

### Helper
original data sample from Nikit
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