# MoE CoLA Training Checklist

Follow these three steps once, then you can launch `sbatch train_moe_cola_slurm.sh` without extra tweaking.

## 1. Create the conda environment

1. Open `LLaMA-Factory/environment.yaml`, set the first `name:` line to your preferred env name (e.g. `cola_llama_factory`), and change the last `prefix:` line to the full path where the env should live (e.g. `/opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3/envs/cola_llama_factory`).
2. Create the environment:
   ```bash
   conda env create -f LLaMA-Factory/environment.yaml
   ```

## 2. Reinstall local packages in editable mode

Run the following inside the repo so the new env uses the local sources:

```bash
conda activate cola_llama_factory
pip uninstall -y peft llamafactory
pip install -e ./peft
pip install -e ./LLaMA-Factory
```

This keeps training and eval in sync with any repo changes.

## 3. Launch the Slurm job

From the repo root:

```bash
sbatch train_moe_cola_slurm.sh
```

The script loads the required modules, activates the conda env, runs MoE CoLA training on 4 GPUs, and triggers `lm_eval` on the produced checkpoints. Monitor via `squeue -u $USER` and check logs under `logs/` plus `${OUTPUT_DIR}/lm_eval`.

# PLEASE EXTEND END DOCUMENT ISSUES or tell me when sth is not working, or you need more details. 
should be easy to fix but i need to know