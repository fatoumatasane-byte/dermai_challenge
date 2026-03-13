import csv

# Lire les vrais labels (separateur ;)
labels = {}
with open('data/test_labels_TRUE.csv', newline='') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        labels[row['image_id']] = int(row['label'])

# Lire la soumission
preds = {}
with open('submissions/Danielle_soumission.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        preds[row['image_id']] = int(row['label'])

# Calculer les metriques
classes = [0, 1]
correct = sum(1 for k in labels if preds.get(k) == labels[k])
acc = correct / len(labels)

tp = {c: sum(1 for k in labels if labels[k]==c and preds.get(k)==c) for c in classes}
fp = {c: sum(1 for k in labels if labels[k]!=c and preds.get(k)==c) for c in classes}
fn = {c: sum(1 for k in labels if labels[k]==c and preds.get(k)!=c) for c in classes}
prec = {c: tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c]>0 else 0 for c in classes}
rec  = {c: tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c]>0 else 0 for c in classes}
f1   = {c: 2*prec[c]*rec[c]/(prec[c]+rec[c]) if prec[c]+rec[c]>0 else 0 for c in classes}
f1_macro   = sum(f1.values()) / 2
prec_macro = sum(prec.values()) / 2
rec_macro  = sum(rec.values()) / 2

print('=' * 45)
print('  DermAI Challenge -- Danielle')
print('=' * 45)
print(f'  Accuracy  : {acc:.4f}')
print(f'  F1 Macro  : {f1_macro:.4f}')
print(f'  Precision : {prec_macro:.4f}')
print(f'  Recall    : {rec_macro:.4f}')
print('=' * 45)
