"""
prepare_competition.py
======================
Run this script ONCE after downloading the Kaggle dataset.

It does 2 things:
  1. Moves all test images into a single flat folder (hides labels from folder names)
  2. Creates all necessary CSV files

BEFORE running this script, your folder should look like:
    data/
    ├── train/
    │   ├── benign/        OK - keep subfolders for train
    │   └── malignant/
    └── test/
        ├── benign/        ← labels visible! must be fixed
        └── malignant/     ← labels visible! must be fixed

AFTER running this script:
    data/
    ├── train/
    │   ├── benign/
    │   └── malignant/
    ├── test/              ← all images mixed in one folder, no subfolders
    │   ├── img_001.jpg
    │   ├── img_002.jpg
    │   └── ...
    ├── train_labels.csv           → publish on GitHub
    ├── test_labels_hidden.csv     → publish on GitHub  (NO labels)
    ├── test_labels_TRUE.csv       → send to professor ONLY, never publish
    └── sample_submission.csv      → publish on GitHub

Usage:
    python prepare_competition.py
"""

import csv
import hashlib
import os
import random
import shutil

SEED     = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SUPPORTED = (".jpg", ".jpeg", ".png", ".bmp")

random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Scan train images and write train_labels.csv
# ─────────────────────────────────────────────────────────────────────────────

def build_train_csv():
    train_dir = os.path.join(DATA_DIR, "train")
    rows = []

    for label, cls in enumerate(["benign", "malignant"]):
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  ⚠️  Not found: {cls_dir}")
            continue
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(SUPPORTED))
        for fname in files:
            rows.append((fname, label))
        print(f"  Train {cls:<12}: {len(files)} images")

    random.shuffle(rows)

    out = os.path.join(DATA_DIR, "train_labels.csv")
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label"])
        for fname, label in rows:
            writer.writerow([fname, label])

    print(f"  ✅ train_labels.csv → {len(rows)} rows\n")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Flatten test folder + write CSVs
# ─────────────────────────────────────────────────────────────────────────────

def flatten_test_and_build_csvs():
    test_dir = os.path.join(DATA_DIR, "test")

    benign_dir    = os.path.join(test_dir, "benign")
    malignant_dir = os.path.join(test_dir, "malignant")

    has_subfolders = os.path.isdir(benign_dir) or os.path.isdir(malignant_dir)

    rows = []   # (filename, true_label)

    if has_subfolders:
        print("  📂 Subfolders detected in test/ → flattening...")

        for label, cls in enumerate(["benign", "malignant"]):
            cls_dir = os.path.join(test_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(SUPPORTED))
            for fname in files:
                src = os.path.join(cls_dir, fname)
                dst = os.path.join(test_dir, fname)
                # Handle duplicate filenames between benign and malignant
                if os.path.exists(dst):
                    name, ext = os.path.splitext(fname)
                    fname = f"{name}_{cls}{ext}"
                    dst   = os.path.join(test_dir, fname)
                shutil.move(src, dst)
                rows.append((fname, label))
            print(f"  Test {cls:<12}: {len(files)} images moved")

        # Remove now-empty subfolders
        for cls in ["benign", "malignant"]:
            cls_dir = os.path.join(test_dir, cls)
            if os.path.isdir(cls_dir) and not os.listdir(cls_dir):
                os.rmdir(cls_dir)
                print(f"  🗑️  Removed empty folder: test/{cls}/")

    else:
        print("  ✅ test/ is already flat (no subfolders)")
        # Read existing hidden CSV if available, else scan folder
        hidden_csv = os.path.join(DATA_DIR, "test_labels_hidden.csv")
        true_csv   = os.path.join(DATA_DIR, "test_labels_TRUE.csv")

        if os.path.exists(true_csv):
            with open(true_csv, newline="") as f:
                for row in csv.DictReader(f):
                    rows.append((row["image_id"], int(row["label"])))
        else:
            print("  ⚠️  No true labels found and test/ is already flat.")
            print("      Cannot rebuild CSVs without knowing the true labels.")
            print("      Make sure you run this script BEFORE flattening manually.")
            return []

    random.shuffle(rows)
    return rows


def write_test_csvs(rows):
    if not rows:
        return

    # TRUE labels — send to professor ONLY
    true_csv = os.path.join(DATA_DIR, "test_labels_TRUE.csv")
    with open(true_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label"])
        for fname, label in rows:
            writer.writerow([fname, label])

    # Hash for verification
    content = open(true_csv, "rb").read()
    sha     = hashlib.sha256(content).hexdigest()
    with open(os.path.join(DATA_DIR, "test_labels_hash.txt"), "w") as f:
        f.write(f"sha256:{sha}\n")
        f.write(f"entries:{len(rows)}\n")

    print(f"\n  🔒 test_labels_TRUE.csv   → {len(rows)} rows  ← SEND TO PROFESSOR ONLY")
    print(f"     SHA-256: {sha[:40]}...")

    # HIDDEN labels — publish on GitHub (no label column)
    hidden_csv = os.path.join(DATA_DIR, "test_labels_hidden.csv")
    with open(hidden_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id"])
        for fname, _ in rows:
            writer.writerow([fname])
    print(f"  📄 test_labels_hidden.csv → {len(rows)} rows  ← PUBLISH ON GITHUB")

    # Sample submission — publish on GitHub
    sample_csv = os.path.join(DATA_DIR, "sample_submission.csv")
    with open(sample_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label"])
        for fname, _ in rows:
            writer.writerow([fname, 0])
    print(f"  📄 sample_submission.csv  → {len(rows)} rows  ← PUBLISH ON GITHUB")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Verify everything looks correct
# ─────────────────────────────────────────────────────────────────────────────

def verify():
    print("\n" + "="*55)
    print("  VERIFICATION")
    print("="*55)

    checks = {
        "data/train/benign/":          os.path.isdir(os.path.join(DATA_DIR, "train", "benign")),
        "data/train/malignant/":       os.path.isdir(os.path.join(DATA_DIR, "train", "malignant")),
        "data/test/  (flat folder)":   os.path.isdir(os.path.join(DATA_DIR, "test")),
        "data/test/benign/  (REMOVED)": not os.path.isdir(os.path.join(DATA_DIR, "test", "benign")),
        "data/test/malignant/  (REMOVED)": not os.path.isdir(os.path.join(DATA_DIR, "test", "malignant")),
        "train_labels.csv":            os.path.exists(os.path.join(DATA_DIR, "train_labels.csv")),
        "test_labels_hidden.csv":      os.path.exists(os.path.join(DATA_DIR, "test_labels_hidden.csv")),
        "test_labels_TRUE.csv":        os.path.exists(os.path.join(DATA_DIR, "test_labels_TRUE.csv")),
        "sample_submission.csv":       os.path.exists(os.path.join(DATA_DIR, "sample_submission.csv")),
    }

    all_ok = True
    for name, ok in checks.items():
        status = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        print(f"  {status}  {name}")

    # Verify hidden CSV has no label column
    hidden_csv = os.path.join(DATA_DIR, "test_labels_hidden.csv")
    if os.path.exists(hidden_csv):
        with open(hidden_csv) as f:
            header = f.readline().strip()
        has_label_col = "label" in header.lower()
        status = "❌  label column found — FIX THIS!" if has_label_col else "✅  no label column"
        if has_label_col:
            all_ok = False
        print(f"\n  Labels hidden in test_labels_hidden.csv : {status}")

    print("\n" + "="*55)
    if all_ok:
        print("""
  ✅ Everything looks correct!

  What to push to GitHub:
    data/train/
    data/test/                  (flat, no subfolders)
    data/train_labels.csv
    data/test_labels_hidden.csv
    data/sample_submission.csv
    scripts/
    baseline/
    docs/
    README.md
    .gitignore

  What to send to the professor by email:
    data/test_labels_TRUE.csv   (NEVER upload this to GitHub)
""")
    else:
        print("  ❌ Some checks failed. Fix the issues above before pushing.")
    print("="*55)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        print(f"❌ data/ folder not found at: {DATA_DIR}")
        print("   Download the Kaggle dataset first and place it in data/")
        exit(1)

    print("="*55)
    print("  DermAI Challenge — Competition Preparation")
    print("="*55)

    print("\n📋 Step 1 — Building train_labels.csv ...")
    build_train_csv()

    print("📋 Step 2 — Flattening test/ folder ...")
    test_rows = flatten_test_and_build_csvs()

    print("\n📋 Step 3 — Writing test CSV files ...")
    write_test_csvs(test_rows)

    print("\n📋 Step 4 — Verifying ...")
    verify()
