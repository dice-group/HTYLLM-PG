#!/bin/bash
#SBATCH --job-name=moe-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one launcher process per node
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2            # 4 GPUs per node
#SBATCH --mem=32GB
#SBATCH --account=hpc-prf-merlin

set -e

source ~/.bashrc
conda activate moe

module load system/CUDA/12.6.0
module load compiler/GCCcore/12.3.0  


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NCCL_DEBUG=INFO


GPUS_PER_NODE=4
NNODES=${SLURM_NNODES}

# pick first node as master
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=6000

export MASTER_ADDR MASTER_PORT

echo "MASTER_ADDR = ${MASTER_ADDR}"
echo "MASTER_PORT = ${MASTER_PORT}"
echo "NNODES      = ${NNODES}"
echo "GPUS/Node   = ${GPUS_PER_NODE}"

deepspeed \
  --num_nodes=${NNODES} \
  --num_gpus=${GPUS_PER_NODE} \
  ../train.py \
    --deepspeed \
    --deepspeed_config ../../ds_config.json \
    --epochs 1 \
    --batch-size 16 \
    --lr 1e-4
