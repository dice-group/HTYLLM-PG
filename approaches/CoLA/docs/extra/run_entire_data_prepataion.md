## Data Prep Pipeline (Manual Run)

Run each stage from its directory so relative paths resolve correctly.

1. **Sampling (FineWeb shards)**
   ```bash
   cd sample_data
   bash run_all_samplers.sh
   ```

2. **Tokenizer extension (per tier)**
   ```bash
   cd data_prep/tokenizer_extension
   bash run_all_tokenizer_extensions.sh
   ```

   After a tier finishes, rebuild its training + tokenizer bundle **before** tokenization:

   ```bash
   export TIER_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/tokenizer_extension/cola_tier1   # or cola_tier2, ...
   rm -rf "${TIER_DIR}/merged_model"
   mkdir -p "${TIER_DIR}/merged_model"
   rsync -a "${TIER_DIR}/initialized_model/" "${TIER_DIR}/merged_model/"
   rsync -a "${TIER_DIR}/extended_tokenizer/" "${TIER_DIR}/merged_model/"
   ```

3. **Tokenize CoLA tiers (base + extended tokenizers)**
   ```bash
   cd distributed_data_processor
   bash tokenize_cola_tiers.sh
   ```

Each script submits the appropriate SLURM jobs and writes logs in its local `logs/` folder (`sample_data/logs`, `data_prep/tokenizer_extension/logs/tokenize_extension`, `distributed_data_processor/logs`). Run stages sequentially; wait for one to finish before starting the next. Combined models for training (merged tokenizer + initialized weights) appear under `/scratch/hpc-prf-merlin/project_data/moe_study/tokenizer_extension/cola_tier*/merged_model`, and tokenized datasets land in `/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_tiers_tokenized/`.
