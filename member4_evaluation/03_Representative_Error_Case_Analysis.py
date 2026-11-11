
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================================================
# STEP 1: Set paths
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SAVE_DIR = ROOT / "member4_evaluation" / "results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

strategies = ["uzeroes", "uones", "uignore"]
names = ["U-Zeroes", "U-Ones", "U-Ignore"]

# Use the thresholds selected on the development set
metrics = pd.read_csv(
    OUTPUTS / "evaluation" / "comparison.csv"
)
thresholds = dict(
    zip(metrics["strategy"], metrics["threshold"])
)

# ==================================================
# STEP 2: Load and aggregate predictions
# ==================================================

all_results = []

for strategy in strategies:

    path = OUTPUTS / strategy / "eval_preds.csv"
    df = pd.read_csv(path)

    # Keep explicit binary labels only
    df = df[df["pleural_effusion_raw"].isin([0, 1])].copy()

    # Aggregate probabilities by patient and study
    grouped = df.groupby(
        ["patient_id", "study_id"],
        as_index=False
    ).agg(
        true_label=("pleural_effusion_raw", "first"),
        prob=("prob", "mean"),
        image_count=("image_path", "count"),
        image_path=("image_path", "first")
    )

    # Select threshold
    threshold = thresholds[strategy]

    grouped["predicted_label"] = (
        grouped["prob"] >= threshold
    ).astype(int)

    # Classify prediction outcomes
    conditions = [
        (grouped["true_label"] == 1) &
        (grouped["predicted_label"] == 1),

        (grouped["true_label"] == 0) &
        (grouped["predicted_label"] == 0),

        (grouped["true_label"] == 0) &
        (grouped["predicted_label"] == 1),

        (grouped["true_label"] == 1) &
        (grouped["predicted_label"] == 0)
    ]

    grouped["outcome"] = np.select(
        conditions,
        ["TP", "TN", "FP", "FN"],
        default="Unknown"
    )

    grouped["strategy"] = strategy
    grouped["threshold"] = threshold

    all_results.append(grouped)

results = pd.concat(all_results, ignore_index=True)

# ==================================================
# STEP 3: Verify evaluation counts
# ==================================================

print("\n=== Evaluation Count Verification ===")

for strategy in strategies:

    subset = results[results["strategy"] == strategy]
    counts = subset["outcome"].value_counts()

    expected = metrics.set_index("strategy").loc[strategy]

    for label in ["tp", "tn", "fp", "fn"]:
        actual = int(counts.get(label.upper(), 0))
        target = int(expected[label])

        if actual != target:
            raise ValueError(
                f"{strategy}: {label} mismatch "
                f"(actual={actual}, expected={target})"
            )

    print(strategy, counts.to_dict())

# ==================================================
# STEP 4: Save representative FP cases
# ==================================================

fp_cases = results[results["outcome"] == "FP"].copy()

# Highest predicted probability among false positives
fp_cases = fp_cases.sort_values(
    ["strategy", "prob"],
    ascending=[True, False]
)

fp_cases.groupby("strategy").head(5).to_csv(
    SAVE_DIR / "false_positive_cases.csv",
    index=False
)

# ==================================================
# STEP 5: Save representative FN cases
# ==================================================

fn_cases = results[results["outcome"] == "FN"].copy()

# Lowest predicted probability among false negatives
fn_cases = fn_cases.sort_values(
    ["strategy", "prob"],
    ascending=[True, True]
)

fn_cases.groupby("strategy").head(5).to_csv(
    SAVE_DIR / "false_negative_cases.csv",
    index=False
)

# ==================================================
# STEP 6: Compare strategy disagreements
# ==================================================

wide = results.pivot(
    index=["patient_id", "study_id"],
    columns="strategy",
    values="predicted_label"
)

# Keep cases present in all three strategies
wide = wide.dropna().astype(int)

disagreements = wide[
    wide.nunique(axis=1) > 1
].copy()

disagreements.to_csv(
    SAVE_DIR / "strategy_disagreements.csv"
)

print("\nStrategy disagreement cases:",
      len(disagreements))

# ==================================================
# STEP 7: Visualize FP and FN counts
# ==================================================

fp_counts = []
fn_counts = []

for strategy in strategies:
    subset = results[results["strategy"] == strategy]

    fp_counts.append(
        (subset["outcome"] == "FP").sum()
    )
    fn_counts.append(
        (subset["outcome"] == "FN").sum()
    )

x = np.arange(len(strategies))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars1 = ax.bar(
    x - width / 2,
    fp_counts,
    width,
    label="False Positive"
)

bars2 = ax.bar(
    x + width / 2,
    fn_counts,
    width,
    label="False Negative"
)

ax.bar_label(bars1, padding=3)
ax.bar_label(bars2, padding=3)

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Number of Cases")
ax.set_title("False Positive and False Negative Comparison")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

save_path = SAVE_DIR / "F3_error_counts.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved files to:", SAVE_DIR)
print("F3_error_counts.png")
print("false_positive_cases.csv")
print("false_negative_cases.csv")
print("strategy_disagreements.csv")
