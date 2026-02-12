sbatch \
    --gres=gpu:h100:1 \
    --time=04:00:00 \
    --partition=gpu \
    --cpus-per-task=4 \
    --mem=160G \
    --output=test.log \
    --wrap "LM_EVAL_LIMIT=500 \
             LM_EVAL_LOG_ROUTER_METRICS=true \
             LM_EVAL_USE_LANG_WRAPPER=true \
             LM_EVAL_TORCH_DTYPE=bf16 \
             LM_EVAL_ATTN_IMPL=flash_attention_2 \
             TOKENIZERS_PARALLELISM=false \
             bash ../lm_eval_checkpoint_adapter_only.sh \
             --checkpoint /scratch/hpc-prf-merlin/project_data/moe_study/multilingual_ablation_200_lang_cola/tier200_full/hydra_hydra-exp-lpr_20260116_094457/checkpoint-10000_adapter \
             --tokenizer meta-llama/Llama-3.1-8B \
             --tasks /scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/configs/lm_eval_tasks_with_flores.txt \
             --batch-size 512 \
             --output-dir tmp_output_single_adapter_eval \
             --lang-mode with_ids \
             --wandb-mode disabled \
             --extra-args \"--include-path /scratch/hpc-prf-merlin/joel/HTYLLM-PG/approaches/CoLA/custom_tasks\""