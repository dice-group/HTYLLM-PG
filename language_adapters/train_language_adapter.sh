#!/bin/bash

#SBATCH --job-name=bloom-adapter-training
#SBATCH --nodes=16  
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --account=hpc-prf-merlin

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node IP: $head_node_ip"
export LOGLEVEL=INFO

source ~/miniconda3/bin/activate icebreaker

TOKENIZED_DATA_DIR="/data/fineweb2_subset_belebele_tokenized_bloom-560m"
MODEL_NAME="bigscience/bloom-560m"
OUTPUT_DIR="/data/joel/bloom560m-belebele-languages"
LOGGING_DIR="$OUTPUT_DIR/logs"

srun torchrun \
  --nnodes=16 \
  --nproc_per_node=1 \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  train_language_adapter.py \
    --tokenized_dir "$TOKENIZED_DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$LOGGING_DIR"
