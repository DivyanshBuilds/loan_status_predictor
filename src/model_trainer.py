'''import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)


def train_model(X_train, X_test, y_train, y_test):

    print("="*50)
    print("RUNNING MODEL TRAINER")
    print("="*50)

    # ── 1. Train Model ────────────────────────────────
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("\n✅ Model trained")

    # ── 2. Stratified Cross Validation ───────────────
    print("\n── Cross Validation (5 Fold) ──")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=['f1', 'precision', 'recall', 'roc_auc']
    )

    print(f"  F1       : {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}")
    print(f"  Precision: {cv_results['test_precision'].mean():.3f} ± {cv_results['test_precision'].std():.3f}")
    print(f"  Recall   : {cv_results['test_recall'].mean():.3f} ± {cv_results['test_recall'].std():.3f}")
    print(f"  ROC-AUC  : {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}")

    # ── 3. Evaluate on Test Set ───────────────────────
    print("\n── Test Set Evaluation ──")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    print(f"  F1       : {f1:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall   : {recall:.3f}")
    print(f"  ROC-AUC  : {roc_auc:.3f}")

    # ── 4. Classification Report ──────────────────────
    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

    # ── 5. Confusion Matrix ───────────────────────────
    print("── Confusion Matrix ──")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted Rejected  Predicted Approved")
    print(f"Actual Rejected       {cm[0][0]:<6}                {cm[0][1]:<6}")
    print(f"Actual Approved       {cm[1][0]:<6}                {cm[1][1]:<6}")

    # ── 6. CV vs Test Gap Check ───────────────────────
    print("\n── Overfitting Check ──")
    cv_f1_mean = cv_results['test_f1'].mean()
    gap = abs(cv_f1_mean - f1)

    if gap < 0.05:
        print(f"✅ CV F1 ({cv_f1_mean:.3f}) vs Test F1 ({f1:.3f}) — gap is small, model is stable")
    elif gap < 0.10:
        print(f"⚠️  CV F1 ({cv_f1_mean:.3f}) vs Test F1 ({f1:.3f}) — small gap, acceptable")
    else:
        print(f"❌ CV F1 ({cv_f1_mean:.3f}) vs Test F1 ({f1:.3f}) — large gap, possible overfitting")

    # ── 7. Save Model ─────────────────────────────────
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("\n✅ Model saved to artifacts/model.pkl")
    print("\n✅ TRAINING COMPLETE\n")

    return model


if __name__ == "__main__":
    from data_validation import validate_data
    from data_transformer import transform_data

    df = validate_data("data/raw/train_u6lujuX_CVtuZ9i.csv")
    X_train, X_test, y_train, y_test = transform_data(df)
    model = train_model(X_train, X_test, y_train, y_test)'''


import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)


def train_model(X_train, X_test, y_train, y_test):

    print("="*50)
    print("RUNNING MODEL TRAINER")
    print("="*50)

    # ── 1. Train Model ────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    print("\n✅ Random Forest model trained")

    # ── 2. Stratified Cross Validation ───────────────
    print("\n── Cross Validation (5 Fold) ──")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=['f1', 'precision', 'recall', 'roc_auc']
    )

    print(f"  F1       : {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}")
    print(f"  Precision: {cv_results['test_precision'].mean():.3f} ± {cv_results['test_precision'].std():.3f}")
    print(f"  Recall   : {cv_results['test_recall'].mean():.3f} ± {cv_results['test_recall'].std():.3f}")
    print(f"  ROC-AUC  : {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}")

    # ── 3. Evaluate on Test Set ───────────────────────
    print("\n── Test Set Evaluation ──")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    print(f"  F1       : {f1:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall   : {recall:.3f}")
    print(f"  ROC-AUC  : {roc_auc:.3f}")

    # ── 4. Classification Report ──────────────────────
    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

    # ── 5. Confusion Matrix ───────────────────────────
    print("── Confusion Matrix ──")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted Rejected  Predicted Approved")
    print(f"Actual Rejected       {cm[0][0]:<6}                {cm[0][1]:<6}")
    print(f"Actual Approved       {cm[1][0]:<6}                {cm[1][1]:<6}")

    # ── 6. CV vs Test Gap Check ───────────────────────
    print("\n── Overfitting Check ──")
    cv_f1_mean = cv_results['test_f1'].mean()
    gap = abs(cv_f1_mean - f1)

    if gap < 0.05:
        print(f"✅ CV F1 ({cv_f1_mean:.3f}) vs Test F1 ({f1:.3f}) — gap is small, model is stable")
    elif gap < 0.10:
        print(f"⚠️  CV F1 ({cv_f1_mean:.3f}) vs Test F1 ({f1:.3f}) — small gap, acceptable")
    else:
        print(f"❌ CV F1 ({cv_f1_mean:.3f}) vs Test F1 ({f1:.3f}) — large gap, possible overfitting")

    # ── 7. Save Model ─────────────────────────────────
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("\n✅ Model saved to artifacts/model.pkl")
    print("\n✅ TRAINING COMPLETE\n")

    return model


if __name__ == "__main__":
    from data_validation import validate_data
    from data_transformer import transform_data

    df = validate_data("data/raw/train_u6lujuX_CVtuZ9i.csv")
    X_train, X_test, y_train, y_test = transform_data(df)
    model = train_model(X_train, X_test, y_train, y_test)