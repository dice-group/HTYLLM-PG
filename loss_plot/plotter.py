import re
import matplotlib.pyplot as plt
import os

# Create a directory for saving plots if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Simulated content from a .out log file (replace this with actual file read)
with open("slurm-23129227.out", "r") as f:
    lines = f.readlines()

# Data containers
steps = []
losses = []
val_losses = []
norms = []
lrs = []
tokens_per_sec = []
dt_ms = []

# Regex patterns
step_loss_pattern = re.compile(
    r"step (\d+)/\d+; loss ([\d.]+); norm ([\d.]+); lr ([\d.]+); tokens/sec ([\d.]+); dt ([\d.]+)ms")
val_loss_pattern = re.compile(r"step (\d+)/\d+; val loss ([\d.]+)")

# Parse the log lines
for line in lines:
    step_loss_match = step_loss_pattern.search(line)
    if step_loss_match:
        step, loss, norm, lr, tps, dt = step_loss_match.groups()
        steps.append(int(step))
        losses.append(float(loss))
        norms.append(float(norm))
        lrs.append(float(lr))
        tokens_per_sec.append(float(tps))
        dt_ms.append(float(dt))
        continue

    val_loss_match = val_loss_pattern.search(line)
    if val_loss_match:
        step, val_loss = val_loss_match.groups()
        steps.append(int(step))
        val_losses.append((int(step), float(val_loss)))

# Extract steps and val_losses as separate lists
val_steps, val_vals = zip(*val_losses) if val_losses else ([], [])

# Plotting and saving
plt.figure(figsize=(12, 6))
plt.plot(steps[:len(losses)], losses, label="Training Loss", linewidth=1)
if val_losses:
    plt.plot(val_steps, val_vals, 'o', label="Validation Loss", color='red')
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/training_validation_loss.png", dpi=300)
plt.close()

# Plot additional metrics
metrics = [("Gradient Norm", norms), ("Learning Rate", lrs),
           ("Tokens/sec", tokens_per_sec), ("Step Time (ms)", dt_ms)]

for title, data in metrics:
    plt.figure(figsize=(12, 4))
    plt.plot(steps[:len(data)], data, linewidth=1)
    plt.title(title)
    plt.xlabel("Step")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{title.lower().replace('/', '_per_')}.png", dpi=300)
    plt.close()

print("All plots saved to 'plots' directory.")