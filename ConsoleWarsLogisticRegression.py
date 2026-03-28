import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report, roc_curve, auc
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH = os.path.join(_SCRIPT_DIR, "Video_Games_Sales_as_at_22_Dec_2016.csv")
_STATIC_DIR = os.path.join(_SCRIPT_DIR, "static")
os.makedirs(_STATIC_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv(_CSV_PATH)

PLATFORM_MAP = {
    "PS": "PlayStation", "PS2": "PlayStation", "PS3": "PlayStation",
    "PS4": "PlayStation", "PSP": "PlayStation", "PSV": "PlayStation",
    "XB": "Xbox", "X360": "Xbox", "XOne": "Xbox",
    "Wii": "Nintendo", "WiiU": "Nintendo", "DS": "Nintendo",
    "3DS": "Nintendo", "GBA": "Nintendo", "GC": "Nintendo",
    "N64": "Nintendo", "SNES": "Nintendo", "NES": "Nintendo",
    "GB": "Nintendo",
}

df["Platform_Family"] = df["Platform"].map(PLATFORM_MAP)
df = df.dropna(subset=["Platform_Family"])

df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")
df = df.dropna(subset=["Critic_Score", "User_Score", "Genre"])

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
genre_dummies = pd.get_dummies(df["Genre"], prefix="genre")
genre_columns = list(genre_dummies.columns)

numeric_features = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales",
                    "Critic_Score", "User_Score"]

X = pd.concat([df[numeric_features].reset_index(drop=True),
               genre_dummies.reset_index(drop=True)], axis=1)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Platform_Family"].values)
class_names = list(label_encoder.classes_)

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------------------------
model = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=42
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
report = classification_report(y_test, y_pred, target_names=class_names)

# ---------------------------------------------------------------------------
# 6. Confusion matrix plot
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
            yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Console Wars Logistic Regression")
fig.tight_layout()
fig.savefig(os.path.join(_STATIC_DIR, "logistic_confusion_matrix.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. ROC curves (One-vs-Rest)
# ---------------------------------------------------------------------------
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
colors = ["#4f6ef7", "#dc2626", "#059669"]

fig, ax = plt.subplots(figsize=(9, 6))
for i, (cls, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Console Wars Logistic Regression (One-vs-Rest)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(_STATIC_DIR, "logistic_roc_curve.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 8. Public helpers
# ---------------------------------------------------------------------------
all_genres = sorted(df["Genre"].unique())
dataset_size = len(df)
train_size = len(X_train)
test_size = len(X_test)


def predict_platform(genre, na_sales, eu_sales, jp_sales, other_sales,
                     critic_score, user_score):
    """Predict platform family given game features."""
    row = {feat: 0.0 for feat in numeric_features}
    row["NA_Sales"] = na_sales
    row["EU_Sales"] = eu_sales
    row["JP_Sales"] = jp_sales
    row["Other_Sales"] = other_sales
    row["Critic_Score"] = critic_score
    row["User_Score"] = user_score

    for col in genre_columns:
        row[col] = 1.0 if col == f"genre_{genre}" else 0.0

    feature_order = numeric_features + genre_columns
    features = np.array([[row[f] for f in feature_order]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    return label_encoder.inverse_transform([prediction])[0], dict(zip(class_names, probabilities))


# ---------------------------------------------------------------------------
# 9. Print stats when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Console Wars — Logistic Regression ===")
    print(f"Dataset rows   : {dataset_size}")
    print(f"Train / Test   : {train_size} / {test_size}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro) : {recall:.4f}")
    print(f"F1 (macro)     : {f1:.4f}")
    print(f"\nClassification Report:\n{report}")
    print("\nExample prediction:")
    pred, probs = predict_platform("Action", 5.0, 3.0, 1.0, 0.5, 80, 7.5)
    print(f"  Action game -> {pred}  {probs}")
