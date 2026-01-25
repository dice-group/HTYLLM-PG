# Project Documentation (Index)

> This index is the entry point for maintainers. Each major area is documented in a dedicated file to make LaTeX/PDF export straightforward.

## Section Files
- **Data preparation**: `docs/data_preparation.md`
- **Model training + implementation**: `docs/model_training_and_implementation.md`
- **Training orchestration (Slurm + listeners)**: `docs/training_orchestration.md`
- **Evaluation + result analysis**: `docs/evaluation_and_analysis.md`
- **Reproducibility + submission checklist**: `docs/reproducibility_and_submission.md`

## Quick Summary
- **Pipeline**: data prep (sampling + clustering + tokenizer extension + tokenization) → training (CoLA/Hydra + LPR) → orchestration (Slurm, checkpoint listeners, eval hooks) → analysis.
- **Tier JSONs**: `tools/two_stage_clustering/*.json` define expert groups + subgroups.
- **Training plan**: `docs/training_plan.md` + `docs/training_runs_plan.csv` (keep in sync with scripts).

## Export Note
- These files are structured with stable headings and minimal nesting for easy conversion to LaTeX/PDF later.

