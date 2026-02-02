import matplotlib.pyplot as plt

# Budgeted tokens by tier
labels = ["High", "Mid", "Low"]
values = [6.23, 4.32, 1.45]  # billions
colors = ["#3b82f6", "#10b981", "#f59e0b"]

total = sum(values)
legend_labels = [
    f"{label}  {((val / total) * 100.0):.2f}%" if total else f"{label}  0.00%"
    for label, val in zip(labels, values)
]

plt.figure(figsize=(4, 4))
plt.pie(
    values,
    labels=None,
    startangle=0,
    colors=colors,
    wedgeprops={"linewidth": 1, "edgecolor": "white"},
)
plt.legend(
    legend_labels,
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
)
plt.tight_layout()
plt.savefig("result_analysis/data_analysis/budgeted_tier_pie.png", dpi=200)
