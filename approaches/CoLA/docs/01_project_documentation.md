# Project Documentation (Index)

> This index is the entry point for maintainers. Each major area is documented in a dedicated file.

## Section Files (Ordered)
- **01 Data preparation**: `docs/02_data_preparation.md`
- **02 Model training + implementation**: `docs/03_model_training_and_implementation.md`
- **03 Training orchestration (Slurm + listeners)**: `docs/04_training_orchestration.md`
- **04 Evaluation + result analysis**: `docs/05_evaluation_and_analysis.md`
- **05 Reproducibility + submission checklist**: `docs/06_reproducibility_and_submission.md`

## Quick Summary
- **Pipeline**: data prep (sampling + clustering + tokenizer extension + tokenization) → training (CoLA/Hydra + LPR) → orchestration (Slurm, checkpoint listeners, eval hooks) → analysis.
- **Tier JSONs**: `tools/two_stage_clustering/*.json` define expert groups + subgroups.

## Export Note
- These files are structured with stable headings and minimal nesting for easy conversion to LaTeX/PDF later.
