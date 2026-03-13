"""
leaderboard_utils.py
====================
Utility functions for evaluating submissions and updating the leaderboard.
Supports both comma (,) and semicolon (;) separators automatically.
"""

import csv
import os
from datetime import datetime

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
LEADERBOARD_CSV  = os.path.join(BASE_DIR, "leaderboard.csv")
TRUE_LABELS_PATH = os.path.join(BASE_DIR, "..", "data", "test_labels_TRUE.csv")


def detect_delimiter(path):
    """Auto-detect delimiter: comma or semicolon."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        first_line = f.readline()
    return ";" if ";" in first_line else ","


def load_true_labels(path=TRUE_LABELS_PATH):
    """Load the hidden true labels — supports , and ; separators."""
    delimiter = detect_delimiter(path)
    labels = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            labels[row["image_id"]] = int(row["label"])
    return labels


def load_submission(path):
    """Load a submission CSV — supports , and ; separators."""
    delimiter = detect_delimiter(path)
    preds = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fields = [f.strip() for f in reader.fieldnames]
        if "image_id" not in fields or "label" not in fields:
            raise ValueError("Submission must have columns: image_id, label")
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            preds[row["image_id"]] = int(row["label"])
    return preds


def compute_metrics(y_true, y_pred):
    """Compute F1 (macro), Accuracy, Precision (macro), Recall (macro)."""
    classes = sorted(set(y_true))
    n = len(y_true)
    if n == 0:
        return {"f1_score": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    precisions, recalls, f1s = [], [], []

    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "f1_score":  round(sum(f1s)        / len(classes), 6),
        "accuracy":  round(accuracy,                        6),
        "precision": round(sum(precisions) / len(classes), 6),
        "recall":    round(sum(recalls)    / len(classes), 6),
    }


def evaluate_submission(submission_path, true_labels_path=TRUE_LABELS_PATH):
    """Evaluate a submission against the true labels."""
    true  = load_true_labels(true_labels_path)
    preds = load_submission(submission_path)

    missing = set(true.keys()) - set(preds.keys())
    extra   = set(preds.keys()) - set(true.keys())

    if missing:
        raise ValueError(f"Missing {len(missing)} predictions. Example: {list(missing)[:3]}")
    if extra:
        raise ValueError(f"Unknown image IDs in submission: {list(extra)[:3]}")

    invalid = [k for k, v in preds.items() if v not in (0, 1)]
    if invalid:
        raise ValueError(f"Invalid label values (must be 0 or 1): {invalid[:3]}")

    y_true = [true[k]  for k in sorted(true.keys())]
    y_pred = [preds[k] for k in sorted(true.keys())]

    return compute_metrics(y_true, y_pred)


def load_leaderboard(path=LEADERBOARD_CSV):
    """Load existing leaderboard entries."""
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("team"):
                continue
            entries.append({
                "team":         row["team"],
                "f1_score":     float(row["f1_score"]),
                "accuracy":     float(row["accuracy"]),
                "precision":    float(row["precision"]),
                "recall":       float(row["recall"]),
                "submitted_at": row["submitted_at"],
            })
    return entries


def save_leaderboard(entries, path=LEADERBOARD_CSV):
    """Save leaderboard entries sorted by F1-Score descending."""
    entries_sorted = sorted(entries, key=lambda x: x["f1_score"], reverse=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["team","f1_score","accuracy","precision","recall","submitted_at"]
        )
        writer.writeheader()
        writer.writerows(entries_sorted)
    return entries_sorted


def update_leaderboard(team_name, metrics, path=LEADERBOARD_CSV):
    """Add or update a team's score — keeps only best score per team."""
    entries = load_leaderboard(path)
    now     = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    new_entry = {
        "team":         team_name,
        "f1_score":     metrics["f1_score"],
        "accuracy":     metrics["accuracy"],
        "precision":    metrics["precision"],
        "recall":       metrics["recall"],
        "submitted_at": now,
    }

    existing = [e for e in entries if e["team"] == team_name]
    if existing:
        best = max(existing, key=lambda x: x["f1_score"])
        if metrics["f1_score"] > best["f1_score"]:
            entries = [e for e in entries if e["team"] != team_name]
            entries.append(new_entry)
            improved = True
        else:
            improved = False
    else:
        entries.append(new_entry)
        improved = True

    return save_leaderboard(entries, path), improved


def print_leaderboard(entries):
    """Print leaderboard to console."""
    print("\n" + "="*65)
    print("  DermAI Challenge — Leaderboard")
    print("="*65)
    print(f"  {'#':>2}  {'Team':<25} {'F1':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7}")
    print("  " + "─"*60)
    medals = ["1.", "2.", "3."]
    for i, e in enumerate(entries, 1):
        medal = ["🥇","🥈","🥉"][i-1] if i <= 3 else f"{i:>2}."
        print(f"  {medal}  {e['team']:<25} {e['f1_score']:>7.4f} {e['accuracy']:>7.4f} "
              f"{e['precision']:>7.4f} {e['recall']:>7.4f}")
    print("="*65 + "\n")
