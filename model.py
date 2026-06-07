# Predictive Modeling Using Machine Learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

# ── 1. Load / Generate Dataset ────────────────────────────────────────────────
np.random.seed(42)
n = 500
X = pd.DataFrame({
    "age":    np.random.randint(20, 65, n),
    "income": np.random.randint(20000, 120000, n),
    "score":  np.random.uniform(300, 850, n),
})
y = ((X["income"] > 60000) & (X["score"] > 600)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 2. Train Models ───────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    results[name] = {"model": clf, "preds": preds, "accuracy": accuracy_score(y_test, preds)}
    print(f"{name}: Accuracy = {results[name]['accuracy']:.4f}")

# ── 3. Confusion Matrices ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, r) in zip(axes, results.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, r["preds"])).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

# ── 4. ROC Curves ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
for name, r in results.items():
    clf = r["model"]
    prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr,tpr):.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(); plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.close()

print("\nPlots saved: confusion_matrices.png, roc_curves.png")
