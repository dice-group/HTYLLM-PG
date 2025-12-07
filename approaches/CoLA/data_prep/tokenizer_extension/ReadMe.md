## Tokenizer extension pipeline
- aims to automate tokenizer_extension strategy at once including coverage analysis, allocation scoring, optional multilingual tokenizer training, and vocabulary extension in one configurable run
- You input a base coverage metrics CSV (or raw shard directory plus tokenizer), per-language token frequency CSVs for extension, and optional JSONL training corpus to build the multilingual tokenizer
- You receive: allocation plan CSV, optionally a trained multilingual tokenizer directory, extended tokenizer directory, coverage CSVs, and comparison tables when requested


- Train (optional): `--train-multilingual` builds the multilingual tokenizer from JSONL shards and saves it.
- Coverage (optional): `--compute-base-coverage` / `--compute-extended-coverage` collect tokenizer metrics.
- Allocation: computes inefficiency scores and writes the token-allocation CSV.
- Extension (optional): `--extend` loads per-language frequency CSVs and updates the base tokenizer.
- Comparison (optional): merge base and extended metrics and save the diff summary.

### SLURM Pipeline
- to run on cluster, use  `tokenize_extension/run_pipeline_slurm.sh`
- Ensure candidate token frequency CSVs exist in `/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/extended_tokenizer/test` (generated via `merge_tokenizer.sh`) before running.

## TODOs
1. adadpt extension due to higher computational demand
2. slurm addition and test
3. briefly docuemtn necessary things
4. end to end pipeline test
5. TODO: list extraction of candidate should also be added instead of assuming it will be given as input


nohup bash tokenizer_extension/run_pipeline_staged.sh tokenizer_extension/configs/llama3.1-8b.yaml > tokenizer_extension/run_llama3.1-8b.out 2>&1 &
disown

nohup bash tokenizer_extension/run_pipeline_staged.sh tokenizer_extension/configs/llama3.2-1b.yaml > tokenizer_extension/run_llama3.2-1b.out 2>&1 &
disown

nohup bash tokenizer_extension/run_pipeline_staged.sh tokenizer_extension/configs/llama3.2-3b.yaml > tokenizer_extension/run_llama3.2-3b.out 2>&1 &
disown
