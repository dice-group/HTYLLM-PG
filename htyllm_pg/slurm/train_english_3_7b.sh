#!/bin/bash
#SBATCH --job-name=moe-multinode
#SBATCH --nodes=4                    
#SBATCH --ntasks-per-node=1           # 1 DeepSpeed launcher per node
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4             # 4 GPUs per node
#SBATCH --mem=32GB
#SBATCH --account=hpc-prf-merlin

set -e

# ---------- Env ----------
source ~/.bashrc

conda activate moe

module load system/CUDA/12.6.0
module load compiler/GCCcore/12.3.0  

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p /scratch/hpc-prf-merlin/luke/.cache/torch_extensions
mkdir -p /scratch/hpc-prf-merlin/luke/.triton/autotune

export TORCH_EXTENSIONS_DIR="/scratch/hpc-prf-merlin/luke/.cache/torch_extensions"

export TRITON_CACHE_DIR="/scratch/hpc-prf-merlin/luke/.triton/autotune"

export XDG_CACHE_HOME="/scratch/hpc-prf-merlin/luke/.cache"


GPUS_PER_NODE=4   

HOSTFILE="${SLURM_SUBMIT_DIR}/hostfile_${SLURM_JOB_ID}"
scontrol show hostnames "${SLURM_JOB_NODELIST}" | while read -r host; do
  echo "${host} slots=${GPUS_PER_NODE}"
done > "${HOSTFILE}"

echo "Hostfile:"
cat "${HOSTFILE}"

MASTER_ADDR=$(head -n 1 "${HOSTFILE}" | awk '{print $1}')
MASTER_PORT=6000

echo "MASTER_ADDR = ${MASTER_ADDR}"
echo "MASTER_PORT = ${MASTER_PORT}"
echo "NNODES      = ${SLURM_NNODES}"

srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 bash -c '
  echo "Node $(hostname) has node_rank=${SLURM_PROCID}"

  deepspeed \
    --hostfile="'${HOSTFILE}'" \
    --no_ssh \
    --node_rank=${SLURM_PROCID} \
    --master_addr="'${MASTER_ADDR}'" \
    --master_port="'${MASTER_PORT}'" \
    htyllm_pg/train.py \
      --deepspeed \
      --deepspeed_config ds_config.json \
      --epochs 5 \
      --batch-size 2 \
      --lr 1e-4 \
      --dim 2048 \
      --depth 24 \
      --heads 16 \
      --dim-head 128 \
      --mlp-dim 8192 \
      --moe-layers 3 7 11 15 19 23 \
      --num-experts 8 \
      --topany-gating-impl "sparse" \
      --use-gradient-checkpointing \
      --use-flash-attention \
      --data-dir /scratch/hpc-prf-merlin/luke/tokenized_english_data
'
