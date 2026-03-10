# Data Description — DermAI Challenge

## Overview

This dataset contains **real dermoscopy images** of skin lesions sourced from the
[ISIC Archive (International Skin Imaging Collaboration)](https://www.isic-archive.com/)
and made available on Kaggle by [fanconic](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign).

The goal is to build a deep learning model that classifies a skin lesion image
as **benign** (non-cancerous) or **malignant** (potentially cancerous).

> **Task**   : Binary image classification  
> **Input**  : RGB dermoscopy image — 224 × 224 pixels  
> **Output** : `0` → Benign · `1` → Malignant

---

## Dataset Statistics

| Split         | Benign | Malignant | Total   |
|---------------|:------:|:---------:|:-------:|
| Train         | 1,440  | 1,197     | **2,637** |
| Test          | 360    | 300       | **660**   |
| **Total**     | **1,800** | **1,497** | **3,297** |

- **Image size** : 224 × 224 pixels (all images are the same size)
- **Colour space** : RGB (3 channels)
- **File format** : JPEG (.jpg)
- **Class balance** : Slightly imbalanced — ~55% benign / ~45% malignant

---

## Folder Structure

```
data/
├── train/
│   ├── benign/               ← 1,440 training images  (label = 0)
│   └── malignant/            ← 1,197 training images  (label = 1)
│
├── test/                     ← 660 images to predict (labels hidden)
│   ├── benign/
│   └── malignant/
│
├── train_labels.csv          ← image filenames + labels for training
├── test_labels_hidden.csv    ← image filenames only (NO labels)
└── sample_submission.csv     ← expected submission format
```

---

## Class Description

### Class 0 — Benign (Non-Cancerous)

A benign skin lesion is harmless and does not invade surrounding tissue.
Common examples include **melanocytic nevi** (ordinary moles).

Visual characteristics:
- **Symmetry** : The lesion looks the same on both halves
- **Borders** : Clear, sharp, and well-defined edges
- **Colour** : Uniform — one consistent shade of brown or tan
- **Texture** : Smooth surface with no irregular patterns
- **Size** : Generally stable over time

### Class 1 — Malignant (Cancerous)

A malignant lesion is potentially cancerous and can spread to other tissues.
The most dangerous form is **melanoma**.

Visual characteristics:
- **Asymmetry** : The two halves of the lesion do not match
- **Borders** : Uneven, ragged, or poorly defined edges
- **Colour** : Multiple shades — dark brown, black, red, or white
- **Texture** : Rough, raised, or irregular surface
- **Size** : May grow rapidly over time

> The ABCD Rule used clinically to assess suspicious lesions:
> Asymmetry · Border irregularity · Colour variation · Diameter > 6mm

---

## CSV File Formats

### train_labels.csv — provided to participants

```
image_id,label
ISIC_0024306.jpg,0
ISIC_0024307.jpg,1
```

### test_labels_hidden.csv — provided to participants (no labels)

```
image_id
ISIC_0034524.jpg
ISIC_0034525.jpg
```

> The test labels are intentionally hidden.
> The professor will use the true labels to evaluate your predictions.

### sample_submission.csv — expected output format

```
image_id,label
ISIC_0034524.jpg,0
ISIC_0034525.jpg,1
```

Submission rules:
- Must contain exactly the 660 images listed in test_labels_hidden.csv
- The label column must be 0 or 1 (integer)
- No missing rows, no duplicate rows

---

## Evaluation Metrics

| Metric       | Formula                 | Priority       |
|--------------|-------------------------|:--------------:|
| F1-Score     | 2 x P x R / (P + R)     | Main metric    |
| Accuracy     | (TP + TN) / Total       | Secondary      |
| Precision    | TP / (TP + FP)          | Reported       |
| Recall       | TP / (TP + FN)          | Reported       |
| ROC AUC      | Area under ROC curve    | Reported       |

F1-Score (Macro) is the main ranking metric. It balances precision and recall
equally across both classes, which is critical in medical diagnosis where both
false positives and false negatives carry significant consequences.

Naive baseline (always predict class 0): F1 = 0.000, Accuracy = 54%

---

## Data Source

- Original source : ISIC Archive — https://www.isic-archive.com
- Kaggle dataset  : https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign
- Image rights    : bound by ISIC Archive terms of use
- Usage           : academic and research purposes only

---

## Suggested Approaches

1. Transfer Learning   : EfficientNet-B0, ResNet-50, DenseNet-121 pretrained on ImageNet
2. Data Augmentation   : random flip, rotation, colour jitter, random crop
3. Class imbalance     : weighted loss function or oversampling of malignant class
4. TTA                 : average predictions across multiple augmented views at test time
5. Ensembling          : combine predictions from multiple models

Monitor Validation F1-Score during training, not just accuracy.

---

## Medical Context

Skin cancer is one of the most common cancers worldwide. Melanoma accounts for
only ~1% of skin cancers but causes the majority of skin cancer deaths.
Early detection through dermoscopy and AI-assisted diagnosis can significantly
increase patient survival rates.

Related datasets:
- HAM10000 : 10,015 dermoscopy images across 7 classes
- ISIC 2018 Challenge : https://challenge.isic-archive.com/landing/2018/
- SIIM-ISIC Melanoma Classification (Kaggle, 33,000+ images)

---

*DermAI Challenge — Deep Learning Course*
*Dataset: Skin Cancer Malignant vs Benign — Kaggle / ISIC Archive*
