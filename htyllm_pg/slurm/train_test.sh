#!/bin/bash
#SBATCH --job-name=moe-multinode
#SBATCH --nodes=2                    
#SBATCH --ntasks-per-node=1           # 1 DeepSpeed launcher per node
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2             # 4 GPUs per node
#SBATCH --mem=32GB
#SBATCH --account=hpc-prf-merlin

set -e

# ---------- Env ----------
source ~/.bashrc

conda activate moe

module load system/CUDA/12.6.0
module load compiler/GCCcore/12.3.0  

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}


GPUS_PER_NODE=2   

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
      --batch-size 4 \
      --lr 1e-4 \
      --topany-gating-impl "opt_mem" \
      --data-dir ../tokenized_subsets/five_representatives_mediods
'
