#!/bin/bash
#SBATCH --job-name=gpt2-sanity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # one DeepSpeed launcher
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1          # single H100 GPU (matches other scripts)
#SBATCH --mem=32GB
#SBATCH --account=hpc-prf-merlin
#SBATCH --qos express

set -e

# ---------- Env ----------
source ~/.bashrc
source /scratch/hpc-prf-merlin/jamil/venv-moe/bin/activate

# Adjust modules to your cluster; mirror existing scripts but stay in user space
module load system/CUDA/12.6.0
module load compiler/GCCcore/12.3.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Per-user caches on scratch (jamil)
export TORCH_EXTENSIONS_DIR="/scratch/hpc-prf-merlin/jamil/.cache/torch_extensions"
export TRITON_CACHE_DIR="/scratch/hpc-prf-merlin/jamil/.cache/triton_autotune"
export XDG_CACHE_HOME="/scratch/hpc-prf-merlin/jamil/.cache"
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}"

# ---------- Run ----------
srun --ntasks=1 --ntasks-per-node=1 bash -c '
  set -euo pipefail
  echo "Node $(hostname) rank=${SLURM_PROCID}"

  deepspeed \
    --num_nodes=1 \
    --num_gpus=1 \
    htyllm_pg/gpt2/train_gpt2.py \
      --deepspeed \
      --deepspeed_config htyllm_pg/gpt2/ds_config_gpt2.json \
      --epochs 1 \
      --batch-size 4 \
      --lr 1e-4 \
      --weight-decay 1e-4 \
      --max-seq-len 2048 \
      --vocab-size 131072 \
      --dim 2048 \
      --depth 24 \
      --heads 16 \
      --dim-head 128 \
      --mlp-dim 8192 \
      --use-gradient-checkpointing \
      --use-flash-attention \
      --data-dir /scratch/hpc-prf-merlin/luke/tokenized_subsets/five_representatives_mediods \
      --checkpoint-dir /scratch/hpc-prf-merlin/jamil/gpt2_sanity_checkpoints \
      --checkpoint-steps 200 \
      --tokenizer-path tokenizer.json
'
