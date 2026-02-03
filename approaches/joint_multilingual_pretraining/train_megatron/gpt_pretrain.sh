#!/bin/bash
#SBATCH --job-name=multinode-gpt2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=32GB
#SBATCH --account=hpc-prf-merlin


module load system/CUDA/12.6.0
module load compiler/GCCcore/12.3.0  
source ~/miniconda3/bin/activate meg

# Runs the 450M parameter model

# Fix for Triton cache NFS issues - use local temp directory
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_PROCID}

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
# MASTER_ADDR=localhost # Removed: torchrun with rdzv_endpoint handles this
# MASTER_PORT=6000 # Removed: torchrun with rdzv_endpoint handles this
NUM_NODES=4
# NODE_RANK=0 # Removed: torchrun handles rank assignment
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "head_node: $head_node"
echo "head_node_ip: $head_node_ip"

CHECKPOINT_PATH=checkpoints/gpt2_legacy #<Specify path>
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

GPT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 1024 
    --num-attention-heads 16 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 8 
    --global-batch-size 6400 
    #--rampup-batch-size 32768 32768 2000000 
    --train-samples   6841663
    --lr-decay-samples 6841663
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-4 
    --lr-decay-style cosine 
    --min-lr 6.0e-5
    --lr-warmup-fraction .06
    --log-throughput
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 1 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 50 
    --eval-interval 100 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# srun will launch torchrun on each allocated node.
# --export=ALL ensures environment variables (like head_node_ip if exported, or SLURM variables) are available.
# The DISTRIBUTED_ARGS are expanded here, and srun passes them to each torchrun instance.
srun --export=ALL torchrun ${DISTRIBUTED_ARGS[@]} ../Megatron-LM/pretrain_gpt.py \
    --ckpt-format torch \
    --use-legacy-models \
    --attention-softmax-in-fp32 \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
    # --load $CHECKPOINT_PATH \
    # --ckpt-convert-format torch \
    # --ckpt-convert-save "checkpoints/torch_format" \