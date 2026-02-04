#!/bin/bash
#SBATCH --job-name=expert_routing_analysis
#SBATCH --output=logs/routing_analysis_%j.out
#SBATCH --error=logs/routing_analysis_%j.err
#SBATCH --time=12:00:00
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
CHECKPOINT="/scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_10_percent/hydra_hydra-exp-lpr_20260108_054502/checkpoint-95000_adapter"
VALIDATION_DATA="/scratch/hpc-prf-merlin/project_data/moe_study/adapter_dataset/cola_200_tier_samples/sharded_samples"
LANGUAGES="acm_Arab,ary_Arab,bel_Cyrl,ceb_Latn,diq_Latn,fao_Latn,glk_Arab,hin_Deva,ind_Latn,kat_Geor,kor_Hang,ltz_Latn,mri_Latn,nno_Latn,pbt_Arab,run_Latn,slv_Latn,sun_Latn,tso_Latn,vls_Latn,ady_Cyrl,arz_Arab,ben_Beng,ces_Latn,div_Thaa,fij_Latn,grc_Grek,hne_Deva,inh_Cyrl,kaz_Cyrl,kpv_Cyrl,lug_Latn,mww_Latn,nob_Latn,plt_Latn,rus_Cyrl,sme_Latn,swe_Latn,tuk_Arab,vro_Latn,aeb_Arab,asm_Beng,bew_Latn,che_Cyrl,dsb_Latn,fin_Latn,gsw_Latn,hrv_Latn,isl_Latn,kbd_Cyrl,lao_Laoo,lus_Latn,mya_Mymr,npi_Deva,pms_Latn,sah_Cyrl,smo_Latn,swh_Latn,tur_Latn,wln_Latn,afr_Latn,ast_Latn,bho_Deva,chv_Cyrl,dzo_Tibt,fra_Latn,guj_Gujr,hsb_Latn,ita_Latn,kha_Latn,lat_Latn,mai_Deva,myv_Cyrl,nrm_Latn,pnb_Arab,san_Deva,sna_Latn,tam_Latn,tyv_Cyrl,wol_Latn,als_Latn,ava_Cyrl,bod_Tibt,ckb_Arab,ekk_Latn,fry_Latn,hat_Latn,hun_Latn,jav_Latn,khk_Cyrl,lez_Cyrl,mal_Latn,nap_Latn,nya_Latn,pol_Latn,scn_Latn,snd_Arab,tat_Cyrl,udm_Cyrl,xho_Latn,amh_Ethi,azb_Arab,bos_Latn,cos_Latn,ell_Grek,fur_Latn,haw_Latn,hye_Armn,jpn_Jpan,khm_Khmr,lim_Latn,mar_Deva,nde_Latn,oci_Latn,por_Latn,sco_Latn,som_Latn,tel_Latn,uig_Arab,ydd_Hebr,apc_Arab,bak_Cyrl,bre_Latn,crh_Cyrl,eng_Latn,gaz_Latn,hbo_Hebr,iba_Latn,kab_Latn,kik_Latn,lin_Latn,mfe_Latn,ndo_Latn,ory_Latn,rmy_Cyrl,sdh_Arab,sot_Latn,tgk_Cyrl,ukr_Cyrl,yor_Latn,arb_Arab,ban_Latn,bul_Cyrl,cym_Latn,epo_Latn,gla_Latn,heb_Hebr,ibo_Latn,kac_Latn,kin_Latn,lit_Latn,mhr_Cyrl,nds_Latn,oss_Cyrl,roh_Latn,shn_Mymr,spa_Latn,tha_Thai,urd_Arab,zea_Latn,arg_Latn,bar_Latn,bxr_Cyrl,dan_Latn,eus_Latn,gle_Latn,hif_Latn,ido_Latn,kal_Latn,kir_Cyrl,lmo_Latn,mkd_Cyrl,new_Deva,pan_Guru,ron_Cyrl,sin_Sinh,srd_Latn,tir_Ethi,uzn_Cyrl,zsm_Arab,ars_Arab,bcl_Latn,cat_Latn,deu_Latn,ewe_Latn,glg_Latn,hil_Latn,ilo_Latn,kan_Knda,kmr_Cyrl,ltg_Latn,mlt_Latn,nld_Latn,pap_Latn,rue_Cyrl,slk_Latn,srp_Cyrl,ton_Latn,vie_Latn,zul_Latn"
NUM_SEQUENCES="10000"
BATCH_SIZE="16"
# Note: --adapter_type, --num_layers, --num_experts are auto-detected from adapter_config.json

# Derived paths
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
VARIANT_NAME=$(basename "$(dirname "$CHECKPOINT")")
OUTPUT_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/extended_analysis/samples_100/${VARIANT_NAME}/${CHECKPOINT_NAME}"
DATA_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/extended_analysis/language_test_sets_cola_200_tier_samples"
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
    --max_sequences 100 \
    --device cuda \
    --use_language_ids \
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
    --output_dir "$OUTPUT_DIR/figures/no_lang_ids" \
    --create_heatmap \
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
