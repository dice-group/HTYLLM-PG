# Reproducibility and Submission Checklist

This section collects environment, install, and artifact notes for maintainers.

---

## 1) Environment and Install
- **Environment file**: `LLaMA-Factory/environment.yaml`
- **Editable installs**:
  - `LLaMA-Factory` and `peft` are installed in editable mode (see `README.md`).

---

## 2) Caches / Runtime Notes
- Training scripts set:
  - `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`
- These avoid NFS cache issues on cluster nodes.

---

## 3) Artifact Checklist (Submission)
- **Source code**: `LLaMA-Factory/src/`, `scripts/`, `tools/`, `docs/`
- **Tier JSONs**: `tools/two_stage_clustering/*.json`
- **Training plan**: `docs/training_plan.md`, `docs/training_runs_plan.csv`
- **Results**: `result_analysis/`

Large directories that may be excluded if needed:
- `outputs/`
- `wandb/`

---

## 4) Testing Notes
- Integration tests live in `tests/integration/` and validate routing + script flags.

