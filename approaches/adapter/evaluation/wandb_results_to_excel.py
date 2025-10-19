import pandas as pd
import wandb

# Initialize API and get the run
api = wandb.Api()
run = api.run("joeldag-paderborn-university/gemma3-4b-pt-yor_latn_wol_latn-adapter/c7r1xcc4")

# Define target metric and _step
metric_cols = [
    "belebele_yor_Latn/acc_none",
    "belebele_wol_Latn/acc_none",
               ]  # You may need to update this metric name if it's different for the new run
columns_to_keep = metric_cols + ["_step"]

# Steps to extract
target_steps = [501, 1002, 1503, 2004, 2505, 3006, 3507, 4008, 4509, 5010, 5511]

# Fetch run history
history = run.history()

# Filter and process
if set(columns_to_keep).issubset(history.columns):
    df = history[columns_to_keep].copy()
    df = df[df["_step"].isin(target_steps)]
    df["row_sum"] = df[metric_cols].sum(axis=1, numeric_only=True)

    print(df)  # Display the filtered result
    df.to_excel("gemma_yor_latn_wol_latn.xlsx", index=False)
else:
    print("Required columns not found in the run history.")
