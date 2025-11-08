#!/bin/bash
#SBATCH --job-name=fw-shard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/shard_%j.out
#SBATCH --error=logs/shard_%j.err
#SBATCH --partition=normal

set -euo pipefail

SOURCE_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples/"
SHARD_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/fw_shards/"
TARGET_SHARD_BYTES=$((512 * 1024 * 1024))

mkdir -p "$(dirname "$SHARD_DIR")" logs

srun python -u shard_corpus.py \
  --source_dir "$SOURCE_DIR" \
  --shard_dir "$SHARD_DIR" \
  --target_shard_bytes "$TARGET_SHARD_BYTES"
