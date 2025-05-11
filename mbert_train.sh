#!/bin/bash

#SBATCH --job-name=multinode-example
#SBATCH --nodes=16   
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --time=11:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --account=hpc-prf-merlin

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

source ~/miniconda3/bin/activate icebreaker

echo "Step 1: Running tokenizer.py..."
python src/data/tokenizer.py 

echo "Step 2: Running mBERT_2.py with multi-GPU support..."
torchrun \
--nnodes 16 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
src/model/mBERT_2.py

echo "Step 3: Evaluating the model..."
python src/model/model_evaluation.py