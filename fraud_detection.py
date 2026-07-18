"""
Credit Card Fraud Detection
----------------------------
Small demo pipeline: load data -> preprocess -> train -> evaluate -> save outputs.

Run from terminal:
    python fraud_detection.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)

# ---------- 1. Load data ----------
DATA_PATH = "creditcard_small.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df["Class"].value_counts())

# ---------- 2. Preprocess ----------
X = df.drop(columns=["Class"])
y = df["Class"]

scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------- 3. Train model ----------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    class_weight="balanced",  # helps with imbalance
    random_state=42,
)
model.fit(X_train, y_train)

# ---------- 4. Evaluate ----------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=3))

auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC Score: {auc:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# ---------- 5. Save plots as output files ----------
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# ---------- 6. Feature importance ----------
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
importances.head(10).plot(kind="barh")
plt.title("Top 10 Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("\nSaved output files: confusion_matrix.png, roc_curve.png, feature_importance.png")
print("Done.")
