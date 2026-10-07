"""
Trains classification models on the placement dataset,
compares them, and saves the best one to model/placement_model.pkl
"""
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/placement_data.csv")

X = df.drop("Placed", axis=1)
y = df["Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
}

results = {}
best_model = None
best_acc = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n{'='*50}\n{name} — Accuracy: {acc:.4f}\n{'='*50}")
    print(classification_report(y_test, preds, target_names=["Not Placed", "Placed"]))

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} ({best_acc:.4f} accuracy)")

# Confusion matrix for best model
os.makedirs("outputs", exist_ok=True)
preds = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Placed", "Placed"], yticklabels=["Not Placed", "Placed"])
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
plt.close()

# Feature importance (Random Forest only)
if best_name == "Random Forest":
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title("Feature Importance")
    plt.savefig("outputs/feature_importance.png", bbox_inches="tight")
    plt.close()

# Save model + scaler
os.makedirs("model", exist_ok=True)
with open("model/placement_model.pkl", "wb") as f:
    pickle.dump({
        "model": best_model,
        "scaler": scaler,
        "columns": list(X.columns),
        "model_name": best_name,
        "accuracy": best_acc
    }, f)

print("\nModel saved to model/placement_model.pkl")
