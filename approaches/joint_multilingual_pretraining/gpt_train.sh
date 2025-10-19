#!/bin/bash

#SBATCH --job-name=multinode-gpt2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --account=hpc-prf-merlin


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

source ~/miniconda3/bin/activate icebreaker

srun torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
src/model/gpt_2_multi_gpu.py --data_path tokenizer/shards_131072/ --micro_batch 8 --tokenizer_path tokenizer/sp_model_131072.model
