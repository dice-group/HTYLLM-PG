#!/bin/bash
# Helper script to resume GPT-2 training from a checkpoint
# Usage: 
#   ./resume_gpt2_training.sh [checkpoint_step]
#   If checkpoint_step is not provided, finds the latest checkpoint automatically
#
# Example:
#   ./resume_gpt2_training.sh 1500
#   ./resume_gpt2_training.sh  # Auto-finds latest

set -e

CHECKPOINT_DIR="/scratch/hpc-prf-merlin/jamil/checkpoints_gpt2_dense"

if [ $# -eq 1 ]; then
    # Use provided checkpoint step
    CHECKPOINT_STEP=$1
else
    # Find latest checkpoint
    echo "Finding latest checkpoint in ${CHECKPOINT_DIR}..."
    
    if [ ! -d "${CHECKPOINT_DIR}" ]; then
        echo "ERROR: Checkpoint directory not found: ${CHECKPOINT_DIR}"
        exit 1
    fi
    
    # Find all step_* directories and extract the highest step number
    LATEST_STEP=$(ls -d ${CHECKPOINT_DIR}/step_* 2>/dev/null | \
        sed 's/.*step_//' | \
        sort -n | \
        tail -1)
    
    if [ -z "${LATEST_STEP}" ]; then
        echo "ERROR: No checkpoints found in ${CHECKPOINT_DIR}"
        exit 1
    fi
    
    CHECKPOINT_STEP=${LATEST_STEP}
    echo "Found latest checkpoint: step_${CHECKPOINT_STEP}"
fi

# Verify checkpoint exists
CHECKPOINT_PATH="${CHECKPOINT_DIR}/step_${CHECKPOINT_STEP}"
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

echo "=========================================="
echo "Resuming GPT-2 Training"
echo "=========================================="
echo "Checkpoint: step_${CHECKPOINT_STEP}"
echo "Path: ${CHECKPOINT_PATH}"
echo "=========================================="
echo ""

# Create a temporary resume script with --load-checkpoint added
RESUME_SCRIPT=$(mktemp)
cat > "${RESUME_SCRIPT}" << 'RESUME_EOF'
#!/bin/bash
#SBATCH --job-name=gpt2-resume
#SBATCH --nodes=2                     # 2 nodes × 4 GPUs = 8 GPUs total
#SBATCH --ntasks-per-node=1           # 1 DeepSpeed launcher per node
#SBATCH --cpus-per-task=4
#SBATCH --time=75:00:00               # 3 days (72h) + 3 hour safety buffer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4             # 4 GPUs per node
#SBATCH --mem=128GB
#SBATCH --account=hpc-prf-merlin

set -e

# ---------- Env ----------
source ~/.bashrc
source /scratch/hpc-prf-merlin/jamil/venv-moe/bin/activate

module load system/CUDA/12.6.0
module load compiler/GCCcore/12.3.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Per-user caches on scratch (jamil)
mkdir -p /scratch/hpc-prf-merlin/jamil/.cache/torch_extensions
mkdir -p /scratch/hpc-prf-merlin/jamil/.triton/autotune

export TORCH_EXTENSIONS_DIR="/scratch/hpc-prf-merlin/jamil/.cache/torch_extensions"
export TRITON_CACHE_DIR="/scratch/hpc-prf-merlin/jamil/.triton/autotune"
export XDG_CACHE_HOME="/scratch/hpc-prf-merlin/jamil/.cache"

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
    htyllm_pg/gpt2/train_gpt2.py \
      --deepspeed \
      --deepspeed_config htyllm_pg/gpt2/ds_config_gpt2.json \
      --epochs 1 \
      # NOTE: --batch-size is deprecated and ignored. Batch size is controlled by ds_config_gpt2.json
      --batch-size 32 \
      --lr 1e-4 \
      --weight-decay 1e-4 \
      --dim 2048 \
      --depth 24 \
      --heads 16 \
      --dim-head 128 \
      --mlp-dim 8192 \
      --vocab-size 131072 \
      --max-seq-len 2048 \
      --use-gradient-checkpointing \
      --use-flash-attention \
      --train-split 1.0 \
      --checkpoint-dir /scratch/hpc-prf-merlin/jamil/checkpoints_gpt2_dense \
      --checkpoint-steps 500 \
      --load-checkpoint CHECKPOINT_STEP_PLACEHOLDER \
      --data-dir /scratch/hpc-prf-merlin/luke/tokenized_multilingual \
      --tokenizer-path tokenizer.json
'
RESUME_EOF

# Replace placeholder with actual checkpoint step
sed -i "s/CHECKPOINT_STEP_PLACEHOLDER/${CHECKPOINT_STEP}/g" "${RESUME_SCRIPT}"

# Submit the job
echo "Submitting resume job..."
JOB_ID=$(sbatch "${RESUME_SCRIPT}" | awk '{print $4}')
echo "Job submitted with ID: ${JOB_ID}"
echo ""
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Cancel with: scancel ${JOB_ID}"
echo ""
echo "Resume script saved to: ${RESUME_SCRIPT}"
echo "(You can delete it after the job starts)"
