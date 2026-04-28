"""Train Extra Trees model on BEED dataset and export artifacts for the API."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

DATA_PATH = "/Users/jackfin/LSTM/BEED_Data.csv"
OUT_DIR = "/Users/jackfin/LSTM/dashboard/backend/models"

df = pd.read_csv(DATA_PATH)
X = df[[f"X{i}" for i in range(1, 17)]].values
y = df["y"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_temp, X_val, y_temp, y_val = train_test_split(X, y_enc, test_size=0.10, stratify=y_enc, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.222, stratify=y_temp, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
X_val_s   = scaler.transform(X_val)

clf = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
f1   = f1_score(y_test, y_pred, average="weighted")
cm   = confusion_matrix(y_test, y_pred).tolist()

# ROC curves (one-vs-rest)
classes = sorted(le.classes_.tolist())
y_test_bin = label_binarize(y_test, classes=list(range(len(classes))))
y_prob = clf.predict_proba(X_test_s)

roc_data = {}
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    roc_data[str(cls)] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": round(roc_auc, 4)
    }

# EEG samples — one per class
samples = {}
for cls in classes:
    idx = np.where(y_enc == cls)[0][0]
    samples[str(cls)] = X[idx].tolist()

# Class distribution
class_counts = {str(cls): int(np.sum(y_enc == cls)) for cls in classes}

# Save artifacts
joblib.dump(clf, f"{OUT_DIR}/extra_trees.joblib")
joblib.dump(scaler, f"{OUT_DIR}/scaler.joblib")
joblib.dump(le, f"{OUT_DIR}/label_encoder.joblib")

artifacts = {
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "f1": round(f1, 4),
    "confusion_matrix": cm,
    "roc": roc_data,
    "eeg_samples": samples,
    "class_counts": class_counts,
    "classes": [str(c) for c in classes],
    "split": {
        "train": len(X_train),
        "test": len(X_test),
        "val": len(X_val)
    }
}

with open(f"{OUT_DIR}/artifacts.json", "w") as f:
    json.dump(artifacts, f)

print(f"Model saved. Test accuracy: {acc:.4f}, F1: {f1:.4f}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
