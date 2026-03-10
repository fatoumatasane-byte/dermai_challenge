# 🏥 DermAI Challenge — Skin Lesion Classification

> Deep Learning Competition · Academic Project

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-green)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Real%20Images-purple)

---

## 🎯 Objective

Build a deep learning model that classifies skin lesion images as **benign** or **malignant**.

| Label | Class         |
|:-----:|---------------|
|  `0`  | **Benign**    |
|  `1`  | **Malignant** |

---

## 📁 Repository Structure

```
dermai-challenge/
│
├── 📂 data/
│   ├── train/
│   │   ├── benign/               ← real training images (class 0)
│   │   └── malignant/            ← real training images (class 1)
│   ├── test/                     ← images to predict (labels hidden)
│   ├── train_labels.csv          ← image names + labels
│   ├── test_labels_hidden.csv    ← image names only (NO labels)
│   └── sample_submission.csv     ← expected submission format
│
├── 📂 scripts/
│   ├── prepare_dataset.py        ← generates CSV files from real images
│   └── evaluate.py               ← professor tool: evaluate submissions
│
├── 📂 baseline/
│   └── baseline_model.py         ← starter model (ResNet18 / MobileNet)
│
├── 📂 docs/
│   └── DATA_DESCRIPTION.md       ← full dataset documentation
│
└── README.md
```

---

## ⚙️ Setup

### 1 — Download the dataset from Kaggle

Go to: **https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign**

Download and place the images following this structure:
```
data/
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

### 2 — Install dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn pillow
```

### 3 — Generate CSV files

```bash
python scripts/prepare_dataset.py
```

This creates:
- `train_labels.csv` → for training
- `test_labels_hidden.csv` → for participants (no true labels)
- `test_labels_TRUE.csv` → **send to professor only, never upload**
- `sample_submission.csv` → submission format example

---

## 🚀 Quick Start (Baseline)

```bash
python baseline/baseline_model.py --data_dir ./data --epochs 10 --model resnet18
```

Your submission file will be saved as `data/my_submission.csv`.

---

## 📊 Dataset

| Split | Benign | Malignant | Total |
|-------|:------:|:---------:|:-----:|
| Train | 1440   | 1197      | ~2637 |
| Test  | 360    | 300       | ~660  |

See [`docs/DATA_DESCRIPTION.md`](docs/DATA_DESCRIPTION.md) for full details.

---

## 📤 Submission Format

Your submission file must follow this exact format:

```csv
image_id,label
2000.jpg,0
2001.jpg,1
...
```

- One row per test image
- `label` must be `0` (benign) or `1` (malignant)
- Image names must match exactly those in `test_labels_hidden.csv`

---

## 🏆 Evaluation

Submissions are ranked by **F1-Score (Macro)**.

| Metric       | Priority     |
|--------------|:------------:|
| **F1-Score** | ⭐ Main      |
| Accuracy     | Secondary    |
| Precision    | —            |
| Recall       | —            |

---

## 📏 Competition Rules

1. ✅ Any deep learning framework is allowed (PyTorch, TensorFlow, Keras…)
2. ✅ Transfer learning is allowed
3. ✅ Data augmentation is allowed
4. ❌ Using the test labels is strictly forbidden
5. ❌ Sharing code between teams is not allowed
6. 📅 Submit your `submission.csv` file before the deadline

---

## 📚 References

- [Kaggle Dataset — Skin Cancer Malignant vs Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)
- [ISIC Skin Lesion Archive](https://www.isic-archive.com/)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
