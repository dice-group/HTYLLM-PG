#!/bin/bash
#SBATCH --job-name=multinode-gpt2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=32GB
#SBATCH --account=hpc-prf-merlin

export PYTHONPATH=../Megatron-LM
module load system/CUDA/12.6.0
module load compiler/GCCcore/12.3.0  
source ~/miniconda3/bin/activate meg

GPUS_PER_NODE=2
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=6002
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
CHECKPOINT_PATH=/scratch/hpc-prf-merlin/htyllm-pg/luke/HTYLLM-PG/checkpoints/gpt2_small_checkpoint
VOCAB_FILE=gpt2_tokenizer/vocab.json
MERGE_FILE=gpt2_tokenizer/merges.txt
DATA_PATH=my-gpt2_text_document
GPT_ARGS="--num-layers 12
--hidden-size 768
--num-attention-heads 12
--seq-length 1024
--max-position-embeddings 1024
--micro-batch-size 12
--global-batch-size 192
--lr 0.0005
--train-iters 1500
--lr-decay-iters 1500
--lr-decay-style cosine
--lr-warmup-iters 100
--weight-decay .1
--adam-beta2 .999
--fp16
--log-interval 10
--save-interval 10
--eval-interval 20
--eval-iters 10
"
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
srun --export=ALL torchrun $DISTRIBUTED_ARGS \
     --rdzv_backend c10d \
     --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        ../Megatron-LM/pretrain_gpt.py \
        --ckpt-format torch \
        --use-legacy-models \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        $GPT_ARGS \
        --attention-softmax-in-fp32 \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        $TENSORBOARD_ARGS
