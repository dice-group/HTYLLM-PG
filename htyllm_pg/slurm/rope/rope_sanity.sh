#!/bin/bash
#SBATCH --job-name=rope-sanity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # one DeepSpeed launcher
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1          # single H100 GPU (matches other scripts)
#SBATCH --mem=32GB
#SBATCH --account=hpc-prf-merlin
#SBATCH --qos express
#SBATCH --chdir=/pc2/users/m/mounzer/HTYLLM-PG

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
    htyllm_pg/train.py \
      --deepspeed \
      --deepspeed_config ds_config.json \
      --epochs 1 \
      --batch-size 4 \
      --max-seq-len 2048 \
      --vocab-size 270000 \
      --dim 256 \
      --depth 4 \
      --heads 8 \
      --dim-head 64 \
      --mlp-dim 1024 \
      --moe-layers 1 3 \
      --num-experts 4 \
      --topany-gating-impl sparse \
      --use-gradient-checkpointing \
      --use-flash-attention \
      --data-dir /scratch/hpc-prf-merlin/luke/tokenized_subsets/five_representatives_mediods \
      --checkpoint-dir /scratch/hpc-prf-merlin/jamil/rope_sanity_checkpoints \
      --checkpoint-steps 200
'
