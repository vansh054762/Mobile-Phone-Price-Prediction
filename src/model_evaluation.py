"""
model_evaluation.py
-------------------
Full evaluation of all 4 models on the real smartphones dataset.
Generates plots saved to models/plots/.

Run: python src/model_evaluation.py
"""

import os, sys
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score,
    roc_curve, accuracy_score,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import get_train_test_split
from model_training      import define_models

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH  = os.path.join(BASE_DIR, "data", "smartphones_specs_clean.csv")
PLOT_DIR   = os.path.join(BASE_DIR, "models", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

CLASSES     = [0, 1, 2, 3]
TIER_NAMES  = ["Budget\n(<₹15k)", "Mid\n(₹15-30k)", "Upper\n(₹30-60k)", "Premium\n(>₹60k)"]
COLORS      = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]


# ── 1. Confusion matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=TIER_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fp = os.path.join(PLOT_DIR, f"cm_{name.replace(' ', '_')}.png")
    plt.savefig(fp, dpi=150); plt.close()
    print(f"   ✔ {fp}")


# ── 2. Accuracy comparison bar chart ─────────────────────────────────────────
def plot_accuracy_comparison(results):
    names = list(results.keys())
    tr    = [results[n]["train_acc"] for n in names]
    te    = [results[n]["test_acc"]  for n in names]
    x, w  = np.arange(len(names)), 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, tr, w, label="Train", color="#4C72B0")
    b2 = ax.bar(x + w/2, te, w, label="Test",  color="#DD8452")
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.legend(); plt.tight_layout()
    fp = os.path.join(PLOT_DIR, "accuracy_comparison.png")
    plt.savefig(fp, dpi=150); plt.close()
    print(f"   ✔ {fp}")


# ── 3. Feature importance ─────────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols, name):
    if not hasattr(model, "feature_importances_"):
        return
    imp  = model.feature_importances_
    idx  = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([feature_cols[i] for i in idx], imp[idx], color="#55A868")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importances — {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fp = os.path.join(PLOT_DIR, f"feature_importance_{name.replace(' ','_')}.png")
    plt.savefig(fp, dpi=150); plt.close()
    print(f"   ✔ {fp}")


# ── 4. ROC curves ─────────────────────────────────────────────────────────────
def plot_roc_curves(results, X_test, y_test):
    y_bin = label_binarize(y_test, classes=CLASSES)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

    for ax, (name, res) in zip(axes.flatten(), results.items()):
        model = res["model"]
        if not hasattr(model, "predict_proba"):
            ax.set_title(name); ax.text(0.4, 0.5, "N/A"); continue
        y_score = model.predict_proba(X_test)
        for ci, (cls, col) in enumerate(zip(CLASSES, COLORS)):
            fpr, tpr, _ = roc_curve(y_bin[:, ci], y_score[:, ci])
            auc = roc_auc_score(y_bin[:, ci], y_score[:, ci])
            ax.plot(fpr, tpr, label=f"Class {cls} (AUC={auc:.2f})", color=col)
        ax.plot([0,1],[0,1],"k--", lw=1)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")

    plt.suptitle("ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fp = os.path.join(PLOT_DIR, "roc_curves.png")
    plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
    print(f"   ✔ {fp}")


# ── 5. Price distribution of real data ───────────────────────────────────────
def plot_price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Class counts
    counts = df["price_range"].value_counts().sort_index()
    axes[0].bar(TIER_NAMES, counts.values, color=COLORS, edgecolor="white")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 2, str(v), ha="center", fontsize=10)
    axes[0].set_title("Price Tier Distribution", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Number of Phones")

    # Price histogram
    axes[1].hist(df["price"], bins=40, color="#4C72B0", edgecolor="white")
    axes[1].set_xlabel("Price (₹)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Actual Price Distribution", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fp = os.path.join(PLOT_DIR, "price_distribution.png")
    plt.savefig(fp, dpi=150); plt.close()
    print(f"   ✔ {fp}")


# ── 6. Top brands by tier ─────────────────────────────────────────────────────
def plot_brand_vs_tier(df):
    top_brands = df["brand"].value_counts().head(12).index
    sub = df[df["brand"].isin(top_brands)]
    ct  = sub.groupby(["brand", "price_range"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, color=COLORS, edgecolor="white")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_ylabel("Number of Phones")
    ax.set_title("Top Brands by Price Tier", fontsize=12, fontweight="bold")
    ax.legend(TIER_NAMES, title="Price Tier", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    fp = os.path.join(PLOT_DIR, "brand_vs_tier.png")
    plt.savefig(fp, dpi=150, bbox_inches="tight"); plt.close()
    print(f"   ✔ {fp}")


# ── Main ──────────────────────────────────────────────────────────────────────
def full_evaluation():
    from data_preprocessing import load_and_clean, encode_and_scale
    from sklearn.model_selection import train_test_split

    print("📂 Loading real dataset …")
    df = load_and_clean(DATA_PATH)

    # Plot dataset-level charts
    print("\n📊 Dataset plots …")
    plot_price_distribution(df)
    plot_brand_vs_tier(df)

    # Encode and split (no stratify — handles tiny classes gracefully)
    X, y, feature_cols, le, scaler = encode_and_scale(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n🔁 Training all models …")
    models  = define_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = {
            "model":     model,
            "train_acc": accuracy_score(y_train, model.predict(X_train)),
            "test_acc":  accuracy_score(y_test,  model.predict(X_test)),
        }

    print("\n📊 Model evaluation plots …")
    plot_accuracy_comparison(results)

    for name, res in results.items():
        print(f"\n── {name}  (Test Acc: {res['test_acc']:.4f}) ──")
        print(classification_report(
            y_test, res["model"].predict(X_test),
            target_names=["Budget","Mid","Upper","Premium"],
            zero_division=0
        ))
        plot_confusion_matrix(res["model"], X_test, y_test, name)
        plot_feature_importance(res["model"], feature_cols, name)

    plot_roc_curves(results, X_test, y_test)

    print(f"\n✅ All plots saved → {PLOT_DIR}")


if __name__ == "__main__":
    full_evaluation()
