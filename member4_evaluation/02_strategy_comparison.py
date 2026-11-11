from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Step 1: Set paths
ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "outputs" / "evaluation" / "comparison.csv"
SAVE_DIR = ROOT / "member4_evaluation" / "results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Step 2: Load results
df = pd.read_csv(INPUT)

strategies = ["uzeroes", "uones", "uignore"]
names = ["U-Zeroes", "U-Ones", "U-Ignore"]

df = df.set_index("strategy").loc[strategies]

# Step 3: Select evaluation metrics
metrics = [
    "auroc",
    "ap",
    "sensitivity",
    "specificity"
]

labels = [
    "AUROC",
    "Average Precision",
    "Sensitivity",
    "Specificity"
]

# Step 4: Plot grouped bars
x = np.arange(len(strategies))
width = 0.2

fig, ax = plt.subplots(figsize=(11, 6))

for i, (metric, label) in enumerate(zip(metrics, labels)):
    values = df[metric].values

    bars = ax.bar(
        x + (i - 1.5) * width,
        values,
        width,
        label=label
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=8
    )

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Metric Value")
ax.set_ylim(0, 1.05)
ax.set_title(
    "Performance Comparison of Three Label Strategies"
)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=4,
    frameon=False
)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

# Step 5: Save figure
save_path = SAVE_DIR / "F2_strategy_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", save_path)

# Step 6: Print comparison table
print("\nPerformance Comparison:")
print(df[metrics].round(4))
