import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
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

# Keep only the four main ESRB ratings
df = df[df["Rating"].isin(["E", "T", "M", "E10+"])]

df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")
df = df.dropna(subset=["Critic_Score", "User_Score", "Genre"])

# ---------------------------------------------------------------------------
# 2. Platform family as a feature
# ---------------------------------------------------------------------------
PLATFORM_MAP = {
    "PS": "PlayStation", "PS2": "PlayStation", "PS3": "PlayStation",
    "PS4": "PlayStation", "PSP": "PlayStation", "PSV": "PlayStation",
    "XB": "Xbox", "X360": "Xbox", "XOne": "Xbox",
    "Wii": "Nintendo", "WiiU": "Nintendo", "DS": "Nintendo",
    "3DS": "Nintendo", "GBA": "Nintendo", "GC": "Nintendo",
    "N64": "Nintendo", "SNES": "Nintendo", "NES": "Nintendo",
    "GB": "Nintendo",
}
df["Platform_Family"] = df["Platform"].map(PLATFORM_MAP).fillna("Other")

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
genre_dummies = pd.get_dummies(df["Genre"], prefix="genre")
genre_columns = list(genre_dummies.columns)

platform_dummies = pd.get_dummies(df["Platform_Family"], prefix="platform")
platform_columns = list(platform_dummies.columns)

numeric_features = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales",
                    "Critic_Score", "User_Score"]

X = pd.concat([
    df[numeric_features].reset_index(drop=True),
    genre_dummies.reset_index(drop=True),
    platform_dummies.reset_index(drop=True),
], axis=1)

all_feature_columns = numeric_features + genre_columns + platform_columns

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Rating"].values)
class_names = list(label_encoder.classes_)

# ---------------------------------------------------------------------------
# 4. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 5. Train KNN model
# ---------------------------------------------------------------------------
K_VALUE = 7
model = KNeighborsClassifier(n_neighbors=K_VALUE)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
report = classification_report(y_test, y_pred, target_names=class_names)

# ---------------------------------------------------------------------------
# 7. Confusion matrix plot
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names,
            yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — KNN (K={K_VALUE}) ESRB Rating Prediction")
fig.tight_layout()
fig.savefig(os.path.join(_STATIC_DIR, "knn_confusion_matrix.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 8. ROC curves (One-vs-Rest)
# ---------------------------------------------------------------------------
y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
colors = ["#4f6ef7", "#dc2626", "#059669", "#d97706"]

fig, ax = plt.subplots(figsize=(9, 6))
for i, (cls, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curves — KNN (K={K_VALUE}) ESRB Rating (One-vs-Rest)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(_STATIC_DIR, "knn_roc_curve.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 9. Public helpers
# ---------------------------------------------------------------------------
all_genres = sorted(df["Genre"].unique())
all_platform_families = sorted(df["Platform_Family"].unique())
dataset_size = len(df)
train_size = len(X_train)
test_size = len(X_test)


def predict_rating(genre, platform_family, na_sales, eu_sales, jp_sales,
                   other_sales, critic_score, user_score):
    """Predict ESRB rating given game features."""
    row = {}
    for feat in numeric_features:
        row[feat] = 0.0
    row["NA_Sales"] = na_sales
    row["EU_Sales"] = eu_sales
    row["JP_Sales"] = jp_sales
    row["Other_Sales"] = other_sales
    row["Critic_Score"] = critic_score
    row["User_Score"] = user_score

    for col in genre_columns:
        row[col] = 1.0 if col == f"genre_{genre}" else 0.0

    for col in platform_columns:
        row[col] = 1.0 if col == f"platform_{platform_family}" else 0.0

    feature_order = numeric_features + genre_columns + platform_columns
    features = np.array([[row[f] for f in feature_order]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    return label_encoder.inverse_transform([prediction])[0], dict(zip(class_names, probabilities))


# ---------------------------------------------------------------------------
# 10. Print stats when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"=== KNN (K={K_VALUE}) — ESRB Rating Prediction ===")
    print(f"Dataset rows   : {dataset_size}")
    print(f"Train / Test   : {train_size} / {test_size}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro) : {recall:.4f}")
    print(f"F1 (macro)     : {f1:.4f}")
    print(f"\nClassification Report:\n{report}")
    print("\nExample prediction:")
    pred, probs = predict_rating("Action", "PlayStation", 5.0, 3.0, 1.0, 0.5, 80, 7.5)
    print(f"  Action/PlayStation -> {pred}  {probs}")
