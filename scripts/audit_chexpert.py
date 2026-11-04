"""Audit CheXpert Small and prepare manifests before patient sampling."""
import argparse
import csv
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

from PIL import Image

FIELDS = ("image_path", "patient_id", "study_id", "pleural_effusion_raw", "split")
LABELS = {"1.0": "1", "0.0": "0", "-1.0": "-1", "1": "1", "0": "0", "-1": "-1", "": ""}
PATH_PATTERN = re.compile(r"(?:CheXpert-v1\.0-small/)?(train|valid)/(patient\d+)/(study\d+)/([^/]+\.jpg)")


def check_image(path):
    if not path.is_file():
        return "missing_image"
    try:
        with Image.open(path) as image:
            image.load()
    except (OSError, ValueError):
        return "unreadable_image"
    return None


def audit(root, source, issues, workers):
    counts = Counter()
    candidates = {}
    conflicts = set()

    def issue(line, path, reason):
        counts[reason] += 1
        issues.writerow((source, line, path, reason))

    with (root / f"{source}.csv").open(newline="") as file:
        for line, row in enumerate(csv.DictReader(file), 2):
            counts["input_rows"] += 1
            if row["Frontal/Lateral"] != "Frontal":
                counts["non_frontal"] += 1
                continue
            raw_path = row["Path"]
            match = PATH_PATTERN.fullmatch(raw_path)
            if not match or match[1] != source:
                issue(line, raw_path, "invalid_path")
                continue
            path = "/".join(match.groups())
            label = LABELS.get(row["Pleural Effusion"].strip())
            if label is None or (source == "valid" and label not in {"0", "1"}):
                issue(line, path, "invalid_label")
                continue
            if path in candidates:
                if candidates[path][1][3] != label:
                    conflicts.add(path)
                    issue(line, path, "conflicting_duplicate")
                else:
                    issue(line, path, "duplicate")
                continue
            split = "final_eval" if source == "valid" else "train_pool"
            candidates[path] = (line, (path, match[2], match[3], label, split))

    eligible = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Batch submissions to keep the executor queue bounded.
        entries = iter(candidates.items())

        while batch := list(islice(entries, 1024)):
            results = pool.map(check_image, (root / path for path, _ in batch))
            for (path, (line, row)), problem in zip(batch, results):
                if path in conflicts:
                    issue(line, path, "excluded_conflicting_path")
                elif problem:
                    issue(line, path, problem)
                else:
                    eligible.append(row)
    counts["eligible_images"] = len(eligible)
    return eligible, dict(counts)


def write_manifest(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(FIELDS)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    pools = {}
    with (args.output_dir / "issues.csv").open("w", newline="") as file:
        issues = csv.writer(file)
        issues.writerow(("source", "csv_line", "image_path", "reason"))
        for source in ("train", "valid"):
            rows, counts = audit(args.data_root, source, issues, args.workers)
            pools[source] = rows
            summary[source] = {
                **counts,
                "patients": len({row[1] for row in rows}),
                "studies": len({row[1:3] for row in rows}),
                "labels": {label: sum(row[3] == label for row in rows)
                           for label in ("1", "0", "-1", "")},
            }
            name = "train_pool.csv" if source == "train" else "final_eval.csv"
            write_manifest(args.output_dir / name, rows)
            print(source, summary[source], flush=True)

    overlap = {row[1] for row in pools["train"]} & {row[1] for row in pools["valid"]}
    summary["patient_overlap"] = len(overlap)
    rng = random.Random(6018)
    sample = []
    for label in ("1", "0", "-1", ""):
        candidates = [row for row in pools["train"] if row[3] == label]
        sample.extend(rng.sample(candidates, min(8, len(candidates))))
    rng.shuffle(sample)
    write_manifest(args.output_dir / "loader_sample.csv", sample)
    with (args.output_dir / "label_distribution.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("source", "raw_label", "images", "percent"))
        for source in ("train", "valid"):
            total = summary[source]["eligible_images"]
            for label, count in summary[source]["labels"].items():
                writer.writerow((source, label, count, round(100 * count / total, 4) if total else 0))
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if overlap:
        raise SystemExit("Patient overlap detected. Review manifests before use.")


if __name__ == "__main__":
    main()
