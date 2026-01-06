REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "${REPO_ROOT}/scripts/tests/checkpoint_listener_local.sh" \
  --watch-dir "${REPO_ROOT}/outputs/local_cola_lpr_2gpu/cola_lpr_20260106_162330" \
  --output-dir "/tmp/lm_eval" \
  --eval-script "${REPO_ROOT}/scripts/tests/lm_eval_checkpoint_local.sh" \
  --tasks "belebele_zul_Latn" \
  --batch-size 1 \
  --tokenizer "hf-internal-testing/tiny-random-LlamaForCausalLM" \
  --wandb-project "htyllm-lm-eval-smoke" \
  --wandb-prefix "cola_lpr_eval" \
  --wandb-mode "online" \
  --once
