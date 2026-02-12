### example uasge

```
python -u result_analysis/eval_analysis.py \
    --train-project dice-nlp/htyllm-adapter-lpr-200_lang_cola \
    --train-group multilingual-ablation-200_lang_cola-20260108_054502 \
    --eval-project dice-nlp/htyllm-adapter-lpr-200_lang_cola_eval \
    --mode with_ids \
    --task-prefix belebele_ \
    --eval-checkpoint 15000 \
    --output-dir result_analysis/paper_eval \
    --api-timeout 60
```