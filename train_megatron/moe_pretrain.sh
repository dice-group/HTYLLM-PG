#!/bin/bash
#SBATCH --job-name=moe-pretrain
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=32GB
#SBATCH --account=hpc-prf-merlin


module load system/CUDA/12.6.0
module load compiler/GCCcore/12.3.0  
source ~/miniconda3/bin/activate meg

# Runs the 450M parameter model

# Fix for Triton cache NFS issues - use local temp directory
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_PROCID}

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
# MASTER_ADDR=localhost # Removed: torchrun with rdzv_endpoint handles this
# MASTER_PORT=6000 # Removed: torchrun with rdzv_endpoint handles this
NUM_NODES=2
# NODE_RANK=0 # Removed: torchrun handles rank assignment
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "head_node: $head_node"
echo "head_node_ip: $head_node_ip"

CHECKPOINT_PATH=checkpoints/moe_pretrain #<Specify path>
TENSORBOARD_LOGS_PATH=logs #<Specify path>
VOCAB_FILE=gpt2_tokenizer/vocab.json #<Specify path to file>/vocab.json
MERGE_FILE=gpt2_tokenizer/merges.txt #<Specify path to file>/merges.txt
DATA_PATH=my-gpt2_text_document #<Specify path and file prefix>_text_document


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
	--rdzv_id $SLURM_JOB_ID
	--rdzv_backend c10d
	--rdzv_endpoint $head_node_ip:29500
)

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 8192
    --num-layers 12
    --hidden-size 512
    --ffn-hidden-size 960
    --num-attention-heads 8
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 4
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)

MOE_ARGS=(
    --num-experts 12
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    # --overlap-param-gather
    # --overlap-grad-reduce
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

TRAINING_ARGS=(
    --micro-batch-size 6
    --global-batch-size 288
    --lr 1e-4
    --train-samples   6841663
    --lr-decay-samples 6841663
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 0
    --clip-grad 1.0
    --bf16
	--log-throughput
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 4
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"}
    )
fi


srun --export=ALL torchrun ${DISTRIBUTED_ARGS[@]} ../Megatron-LM/pretrain_gpt.py \
	--ckpt-format torch \
	--use-legacy-models \
    --attention-softmax-in-fp32 \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}