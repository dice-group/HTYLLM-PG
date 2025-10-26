#!/bin/bash
#SBATCH --job-name=moe-pretraining
#SBATCH --nodes=8                   
#SBATCH --ntasks-per-node=8          
#SBATCH --cpus-per-task=16           
#SBATCH --gres=gpu:h100:8                 
#SBATCH --time=00:15:00              
#SBATCH --partition=gpu              
#SBATCH --output=logs/moe_%j.out
#SBATCH --error=logs/moe_%j.err

module load system/CUDA/12.6.0
module load compiler/GCCcore/12.3.0  


source .venv/bin/activate

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# Get master node address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

MODEL_SIZE="8x7B"  #
TP=1               # Tensor parallelism
PP=4               # Pipeline parallelism
EP=8               # Expert parallelism  -> must divide num_experts
CP=1               # Context parallelism
NUM_EXPERTS=8

DATA_PATH="data/fineweb2_subset"
TOKENIZER_PATH="tokenizer/sp_model_131072/sp_model_131072.model"
CHECKPOINT_DIR="checkpoints/fineweb2_subset"


uv run srun --nodes=$NNODES \
     --ntasks-per-node=$GPUS_PER_NODE \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     bash -c "
     torchrun \
         --nnodes=$NNODES \
         --nproc_per_node=$GPUS_PER_NODE \
         --rdzv_id=$SLURM_JOB_ID \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         pretrain_gpt.py \
         --tensor-model-parallel-size $TP \
         --pipeline-model-parallel-size $PP \
         --expert-model-parallel-size $EP \
         --context-parallel-size $CP \
         --sequence-parallel \
         --num-experts $NUM_EXPERTS \
         --moe-grouped-gemm \
         --moe-router-topk 2 \
         --moe-aux-loss-coeff 0.01 \
         --num-layers 32 \
         --hidden-size 4096 \
         --ffn-hidden-size 14336 \
         --num-attention-heads 32 \
         --seq-length 4096 \
         --max-position-embeddings 32768 \
         --micro-batch-size 1 \
         --global-batch-size 1024 \
         --train-iters 100000 \
         --lr 3e-4 \
         --min-lr 3e-5 \
         --lr-decay-style cosine \
         --lr-warmup-iters 2000 \
         --weight-decay 0.1 \
         --clip-grad 1.0 \
         --bf16 \
         --use-distributed-optimizer \
         --overlap-grad-reduce \
         --overlap-param-gather \
         --attention-backend sdpa \
         --data-path $DATA_PATH \
         --tokenizer-type HuggingFaceTokenizer \
         --tokenizer-model $TOKENIZER_PATH \
         --save $CHECKPOINT_DIR \
         --load $CHECKPOINT_DIR \
         --save-interval 1000 \
         --eval-interval 500 \
         --eval-iters 100 \
         --log-interval 10
     "