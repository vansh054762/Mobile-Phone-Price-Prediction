"""
model_training.py
-----------------
Trains 4 ML models on the real smartphones dataset and saves the best one.
"""

import os, sys, joblib
import numpy as np
from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics        import accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import get_train_test_split

BASE_DIR   = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH  = os.path.join(BASE_DIR, "data", "smartphones_specs_clean.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


def define_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=1.0, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_split=8, min_samples_leaf=4,
            random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=6,
            min_samples_leaf=3, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.07, max_depth=4,
            min_samples_leaf=4, subsample=0.8,
            random_state=RANDOM_STATE),
    }


def train_all():
    print(f"📂 Loading: {DATA_PATH}")
    X_train, X_test, y_train, y_test, feature_cols, le, scaler = \
        get_train_test_split(DATA_PATH)

    print(f"   Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}\n")
    models  = define_models()
    results = {}

    print(f"{'Model':<25} {'Train Acc':>10} {'Test Acc':>10}")
    print("─" * 48)

    for name, model in models.items():
        model.fit(X_train, y_train)
        tr = accuracy_score(y_train, model.predict(X_train))
        te = accuracy_score(y_test,  model.predict(X_test))
        results[name] = dict(model=model, train_acc=tr, test_acc=te,
                              X_test=X_test, y_test=y_test)
        print(f"{name:<25} {tr:>10.4f} {te:>10.4f}")

    best_name = max(results, key=lambda k: results[k]["test_acc"])
    print(f"\n🏆 Best: {best_name}  (Test Acc = {results[best_name]['test_acc']:.4f})")

    bundle = {
        "model":         results[best_name]["model"],
        "model_name":    best_name,
        "scaler":        scaler,
        "label_encoder": le,
        "feature_cols":  feature_cols,
    }
    save_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(bundle, save_path)
    print(f"✅ Saved → {save_path}")
    return results, bundle


if __name__ == "__main__":
    train_all()
