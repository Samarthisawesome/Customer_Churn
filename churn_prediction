import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    RocCurveDisplay
)

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")  # place CSV in same folder

# ─────────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df = df.drop("customerID", axis=1)

# ─────────────────────────────────────────
# 3. ENCODE TARGET & CATEGORICAL FEATURES
# ─────────────────────────────────────────
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

# ─────────────────────────────────────────
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Exploratory Data Analysis", fontsize=16)

sns.countplot(x="Churn", data=df, ax=axes[0, 0])
axes[0, 0].set_title("Churn Distribution")

sns.boxplot(x="Churn", y="tenure", data=df, ax=axes[0, 1])
axes[0, 1].set_title("Tenure vs Churn")

sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=axes[0, 2])
axes[0, 2].set_title("Monthly Charges vs Churn")

contract_cols = [c for c in df.columns if "Contract" in c]
if contract_cols:
    df_plot = df[contract_cols + ["Churn"]].copy()
    df_plot.groupby("Churn")[contract_cols].mean().T.plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title("Contract Type vs Churn")
    axes[1, 0].tick_params(axis="x", rotation=30)

sns.heatmap(
    df.corr()[["Churn"]].sort_values("Churn", ascending=False).head(15),
    annot=True, cmap="coolwarm", ax=axes[1, 1], cbar=False
)
axes[1, 1].set_title("Top Correlations with Churn")

axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print(" EDA plots saved to eda_plots.png")

# ─────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ─────────────────────────────────────────
# 6. SCALE FEATURES
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─────────────────────────────────────────
# 7. LOGISTIC REGRESSION (baseline)
# ─────────────────────────────────────────
print("\n⏳ Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_cv   = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring="roc_auc")

print("=" * 50)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 50)
print(f"Accuracy           : {accuracy_score(y_test, lr_pred):.4f}")
print(f"ROC-AUC            : {roc_auc_score(y_test, lr_prob):.4f}")
print(f"CV ROC-AUC (5-fold): {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

# ─────────────────────────────────────────
# 8. RANDOM FOREST — DEFAULT (before tuning)
# ─────────────────────────────────────────
print("\n⏳ Training Random Forest (default)...")
rf_default = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
rf_default.fit(X_train_scaled, y_train)

rf_default_pred = rf_default.predict(X_test_scaled)
rf_default_prob = rf_default.predict_proba(X_test_scaled)[:, 1]

print("=" * 50)
print("RANDOM FOREST — DEFAULT")
print("=" * 50)
print(f"Accuracy : {accuracy_score(y_test, rf_default_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, rf_default_prob):.4f}")

# ─────────────────────────────────────────
# 9. HYPERPARAMETER TUNING — RandomizedSearchCV
# ─────────────────────────────────────────
print("\n Running RandomizedSearchCV (this may take ~1–2 mins)...")

param_grid = {
    "n_estimators"      : [100, 200, 300, 500],
    "max_depth"         : [None, 5, 10, 20, 30],
    "min_samples_split" : [2, 5, 10],
    "min_samples_leaf"  : [1, 2, 4],
    "max_features"      : ["sqrt", "log2"],
    "bootstrap"         : [True, False]
}

rf_search = RandomizedSearchCV(
    estimator           = RandomForestClassifier(class_weight="balanced", random_state=42),
    param_distributions = param_grid,
    n_iter              = 50,        # tries 50 random combinations
    scoring             = "roc_auc", # optimize for ROC-AUC
    cv                  = 5,         # 5-fold cross-validation
    verbose             = 1,
    random_state        = 42,
    n_jobs              = -1         # use all CPU cores
)

rf_search.fit(X_train_scaled, y_train)

print(f"\nBest Parameters Found:\n{rf_search.best_params_}")
print(f" Best CV ROC-AUC: {rf_search.best_score_:.4f}")

# ─────────────────────────────────────────
# 10. EVALUATE TUNED MODEL
# ─────────────────────────────────────────
best_rf = rf_search.best_estimator_

rf_pred = best_rf.predict(X_test_scaled)
rf_prob = best_rf.predict_proba(X_test_scaled)[:, 1]
rf_cv   = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring="roc_auc")

print("\n" + "=" * 50)
print("RANDOM FOREST — TUNED")
print("=" * 50)
print(f"Accuracy           : {accuracy_score(y_test, rf_pred):.4f}")
print(f"ROC-AUC            : {roc_auc_score(y_test, rf_prob):.4f}")
print(f"CV ROC-AUC (5-fold): {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# Before vs After summary
print("\n" + "=" * 50)
print("TUNING IMPROVEMENT SUMMARY")
print("=" * 50)
before = roc_auc_score(y_test, rf_default_prob)
after  = roc_auc_score(y_test, rf_prob)
print(f"ROC-AUC Before Tuning : {before:.4f}")
print(f"ROC-AUC After Tuning  : {after:.4f}")
print(f"Improvement           : +{after - before:.4f}")

# ─────────────────────────────────────────
# 11. VISUALIZATION
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Tuned Model Results", fontsize=16)

# Confusion Matrix — LR
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Logistic Regression\nConfusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Confusion Matrix — Tuned RF
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Random Forest (Tuned)\nConfusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# ROC Curves — both models overlaid
RocCurveDisplay.from_predictions(y_test, lr_prob, name="Logistic Regression", ax=axes[2])
RocCurveDisplay.from_predictions(y_test, rf_prob, name="Random Forest (Tuned)", ax=axes[2])
axes[2].set_title("ROC Curve Comparison")
axes[2].plot([0, 1], [0, 1], "k--", label="Random Chance")
axes[2].legend()

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Model comparison plots saved to model_comparison.png")

# Feature Importance
fig, ax = plt.subplots(figsize=(8, 6))
rf_importance = pd.Series(best_rf.feature_importances_, index=X.columns)
rf_importance.sort_values().tail(10).plot(kind="barh", color="steelblue", ax=ax)
ax.set_title("Top 10 Features (Tuned Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Feature importance plot saved to feature_importance.png")

# ─────────────────────────────────────────
# 12. SAVE MODEL, SCALER & FEATURE COLUMNS
# ─────────────────────────────────────────
pickle.dump(best_rf, open("churn_model.pkl", "wb"))
pickle.dump(scaler,  open("scaler.pkl", "wb"))
pickle.dump(list(X.columns), open("feature_columns.pkl", "wb"))

print("\nTuned model, scaler, and feature columns saved!")
print("Files: churn_model.pkl | scaler.pkl | feature_columns.pkl")
