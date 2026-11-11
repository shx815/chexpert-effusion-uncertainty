from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Step 1: Set project paths
ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "outputs" / "evaluation" / "comparison.csv"
SAVE_DIR = ROOT / "member4_evaluation" / "results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Step 2: Load evaluation results
df = pd.read_csv(INPUT)

print("Evaluation Results:")
print(df.to_string(index=False))

# Step 3: Draw confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

strategies = ["uzeroes", "uones", "uignore"]
titles = ["U-Zeroes", "U-Ones", "U-Ignore"]

for ax, strategy, title in zip(axes, strategies, titles):

    row = df[df["strategy"] == strategy].iloc[0]

    tn = int(row["tn"])
    fp = int(row["fp"])
    fn = int(row["fn"])
    tp = int(row["tp"])

    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d"
    )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.suptitle(
    "Confusion Matrix Comparison",
    fontsize=16
)

plt.tight_layout()

# Step 4: Save figure
save_path = SAVE_DIR / "F1_confusion_matrices.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved figure:")
print(save_path)
