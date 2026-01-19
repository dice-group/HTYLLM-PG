#!/bin/bash
#SBATCH --job-name=expert_routing_analysis
#SBATCH --output=logs/routing_analysis_%j.out
#SBATCH --error=logs/routing_analysis_%j.err
#SBATCH --time=35:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu

# Expert Routing Analysis - Slurm Version
# This script is optimized for HPC clusters with Slurm workload manager

set -e  # Exit on error

echo "========================================="
echo "Expert Routing Analysis (Slurm)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo ""


set -euo pipefail

module purge
module load toolchain/foss/2024a
module load system/CUDA/12.6.0
module load lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

# Use shared scratch HF cache
export HF_HOME=/scratch/hpc-prf-merlin/shared_cache/huggingface/hub
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

source /opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate hydralora_llama_factory
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
# Disabled: torchrun handles GPU assignment automatically
# if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
#   export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
# fi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count());"


# Configuration from command line or defaults
BASE_MODEL="meta-llama/Llama-3.1-8B"
CHECKPOINT="/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/cola_colaexp-hard_20260108_054502/checkpoint-60000_adapter"
VALIDATION_DATA="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/sharded_samples"
LANGUAGES="wbm_Latn,tel_Telu,fas_Arab,crl_Cans,abk_Cyrl,kru_Deva,ksw_Mymr,ekk_Latn,lao_Laoo,uzs_Arab,srp_Cyrl,kat_Geor,ory_Orya,hbo_Hebr,mhr_Cyrl,mni_Beng,moc_Latn,toc_Latn,agg_Latn,amr_Latn,sat_Olck,esi_Latn,fuv_Latn,tzm_Tfng,enx_Latn,gcf_Latn,pnt_Grek,pui_Latn,naq_Latn,kxm_Thai,guu_Latn,sah_Cyrl,spm_Latn,wuu_Hani,nas_Latn,dzo_Tibt,yuj_Latn,gnw_Latn,dty_Deva,jbo_Latn,wsg_Telu,nwx_Deva,spa_Latn,poi_Latn,crx_Latn,xav_Latn,tcy_Knda,gbi_Latn,mej_Latn,hto_Latn,gym_Latn,kaq_Latn,tat_Latn,dik_Latn,naf_Latn,liv_Latn,avt_Latn,miq_Latn,orv_Cyrl,azg_Latn,caa_Latn,jiv_Latn,asm_Beng,tig_Ethi,cpu_Latn,syc_Syrc,tkr_Cyrl,zsm_Latn,lsi_Latn,xal_Cyrl,tel_Latn,hin_Deva,dhg_Latn,cho_Latn,nab_Latn,ium_Latn,bon_Latn,ayo_Latn,cbi_Latn,ncj_Latn,ake_Latn,xnn_Latn,auc_Latn,ang_Latn,quz_Latn,emp_Latn,fin_Latn,guh_Latn,gum_Latn,jaa_Latn,sdh_Arab,wlv_Latn,crk_Latn,knv_Latn,tuo_Latn,slv_Latn,iws_Latn,arz_Arab,hae_Latn,myb_Latn,bug_Latn,koi_Cyrl,aey_Latn,atd_Latn,srd_Latn,wmt_Latn,gqr_Latn,kyf_Latn,sma_Latn,ige_Latn,ctp_Latn,mti_Latn,otw_Latn,xmm_Latn,chq_Latn,lbb_Latn,azj_Cyrl,csy_Latn,pxm_Latn,mse_Latn,ots_Latn,kxw_Latn,lob_Latn,njo_Latn,yua_Latn,kgk_Latn,hig_Latn,kyu_Latn,rtm_Latn,ile_Latn,hin_Latn,sbd_Latn,mww_Latn,hch_Latn,maf_Latn,snd_Deva,tee_Latn,toi_Latn,yut_Latn,swp_Latn,mqj_Latn,ubu_Latn,rus_Cyrl,nin_Latn,ksc_Latn,keo_Latn,esu_Latn,amk_Latn,bqp_Latn,laj_Latn,ibg_Latn,kck_Latn,poy_Latn,tay_Latn,rmn_Latn,cym_Latn,ojb_Cans,bgs_Latn,oku_Latn,mcb_Latn,cub_Latn,whg_Latn,bov_Latn,prf_Latn,hnn_Latn,lgl_Latn,kpw_Latn,dnj_Latn,cbr_Latn,jav_Latn,nba_Latn,gso_Latn,ntp_Latn,gng_Latn,prg_Latn,gwr_Latn,gmv_Latn,cpc_Latn,nuj_Latn,pma_Latn,acn_Latn,nii_Latn,rgu_Latn,zpq_Latn,mck_Latn,uvh_Latn,wew_Latn,bbr_Latn,bwq_Latn,gej_Latn,pam_Latn,nya_Latn,tso_Latn,mau_Latn,tue_Latn,bku_Latn,mie_Latn,ogo_Latn,ben_Latn,english"
NUM_SEQUENCES="10000"
BATCH_SIZE="16"
# Note: --adapter_type, --num_layers, --num_experts are auto-detected from adapter_config.json

# Derived paths
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
OUTPUT_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/extended_analysis/${CHECKPOINT_NAME}"
DATA_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/extended_analysis/language_test_sets"
LOGS_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/extended_analysis/logs"

# Create logs directory
mkdir -p "$LOGS_DIR"

echo "Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  Checkpoint: $CHECKPOINT"
echo "  Languages: $LANGUAGES"
echo "  Batch Size: $BATCH_SIZE"
echo "  Output: $OUTPUT_DIR"
echo "  (adapter_type, num_layers, num_experts auto-detected)"
echo ""

# Step 1: Prepare test data (skip if already exists)
if [ ! -d "$DATA_DIR" ]; then
    echo "[1/5] Preparing language test datasets..."
    srun python tool/prepare_language_datasets.py \
        --validation_data "$VALIDATION_DATA" \
        --languages "$LANGUAGES" \
        --num_sequences "$NUM_SEQUENCES" \
        --output_dir "$DATA_DIR" \
        2>&1 | tee "$LOGS_DIR/step1_prepare_data_${SLURM_JOB_ID}.log"
    echo ""
else
    echo "[1/5] Skipping data preparation (already exists)"
    echo ""
fi

# Step 2: Analyze routing (adapter_type, num_layers, num_experts auto-detected)
echo "[2/5] Running expert routing analysis..."
srun python tool/analyze_expert_routing.py \
    --base_model "$BASE_MODEL" \
    --adapter_checkpoint "$CHECKPOINT" \
    --test_data "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_sequences 100\
    --device cuda \
    --use_language_ids
    2>&1 | tee "$LOGS_DIR/step2_analyze_${SLURM_JOB_ID}.log"
echo ""

# Step 3: Normalize data
echo "[3/5] Applying layer-wise normalization..."
srun python tool/process_routing_data.py \
    --input "$OUTPUT_DIR/routing_matrix.npz" \
    --output "$OUTPUT_DIR/routing_matrix_normalized.npz" \
    2>&1 | tee "$LOGS_DIR/step3_normalize_${SLURM_JOB_ID}.log"
echo ""

# Step 4: Generate visualizations
echo "[4/5] Creating visualizations..."
srun python tool/visualize_expert_routing.py \
    --routing_data "$OUTPUT_DIR/routing_matrix_normalized.npz" \
    --language_families ./config/language_families.json \
    --output_dir "$OUTPUT_DIR/figures" \
    --create_all \
    2>&1 | tee "$LOGS_DIR/step4_visualize_${SLURM_JOB_ID}.log"
echo ""

# Step 5: Generate report
echo "[5/5] Generating analysis report..."
srun python tool/generate_analysis_report.py \
    --routing_data "$OUTPUT_DIR/routing_matrix_normalized.npz" \
    --language_families ./config/language_families.json \
    --figures_dir "$OUTPUT_DIR/figures" \
    --output "$OUTPUT_DIR/report.md" \
    2>&1 | tee "$LOGS_DIR/step5_report_${SLURM_JOB_ID}.log"
echo ""

echo "========================================="
echo "Analysis Complete!"
echo "========================================="
echo "Finished at: $(date)"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View report: $OUTPUT_DIR/report.md"
echo "View figures:"
echo "  - Heatmap: $OUTPUT_DIR/figures/routing_heatmap.png"
echo "  - t-SNE: $OUTPUT_DIR/figures/tsne_clustering.png"
echo "  - Entropy: $OUTPUT_DIR/figures/layer_entropy.png"
echo ""
echo "Job completed successfully!"
