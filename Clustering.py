import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH = os.path.join(_SCRIPT_DIR, "Video_Games_Sales_as_at_22_Dec_2016.csv")
_STATIC_DIR = os.path.join(_SCRIPT_DIR, "static")
os.makedirs(_STATIC_DIR, exist_ok=True)

_CLUSTER_COLORS = ["#4f6ef7", "#dc2626", "#d97706"]

# ---------------------------------------------------------------------------
# 1. Load and prepare base dataset
# ---------------------------------------------------------------------------
_df = pd.read_csv(_CSV_PATH)
_df["User_Score"] = pd.to_numeric(_df["User_Score"], errors="coerce")
_df = _df.dropna(subset=["Name", "Critic_Score", "User_Score", "Global_Sales"])
_df = _df[(_df["Critic_Score"] > 0) & (_df["User_Score"] > 0) & (_df["Global_Sales"] > 0)]
_df = _df.reset_index(drop=True)


# ===========================================================================
# PART 1  ---  Manual K-Means Simulation (100 records, 2 features)
# Features: Critic_Score (0-100), Global_Sales (millions of units)
# ===========================================================================

MANUAL_FEATURES = ["Critic_Score", "Global_Sales"]

# Manually chosen initial centroids — spread across the data range
INITIAL_CENTROIDS = [
    {"critic_score": 55.0, "global_sales": 0.20},   # Low-critic / low-sales
    {"critic_score": 75.0, "global_sales": 1.50},   # Mid-critic / mid-sales
    {"critic_score": 90.0, "global_sales": 6.00},   # High-critic / high-sales
]


def getManualDataset():
    """Return 100 games with Critic_Score and Global_Sales (Part 1 dataset)."""
    subset = _df.sample(n=100, random_state=42).reset_index(drop=True)
    records = []
    for i, row in subset.iterrows():
        records.append({
            "id": i + 1,
            "name": str(row["Name"])[:40],
            "critic_score": float(row["Critic_Score"]),
            "global_sales": round(float(row["Global_Sales"]), 2),
        })
    return records


def euclidean_distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def runManualKMeans():
    """Run 3 manual K-Means iterations on the 100-record dataset."""
    data = getManualDataset()
    X = np.array([[d["critic_score"], d["global_sales"]] for d in data])

    centroids = np.array([
        [c["critic_score"], c["global_sales"]] for c in INITIAL_CENTROIDS
    ])
    k = len(centroids)
    iterations = []

    for it in range(3):
        iter_centroids_in = centroids.copy()

        # Step A. Distances from every point to every centroid
        distances = np.zeros((len(X), k))
        for i, point in enumerate(X):
            for j, c in enumerate(iter_centroids_in):
                distances[i, j] = euclidean_distance(point, c)

        # Step B. Assign each point to the nearest centroid
        assignments = np.argmin(distances, axis=1)

        # Step C. Variance (Within-Cluster Sum of Squares)
        wcss = float(np.sum(np.min(distances, axis=1) ** 2))

        # Step D. Recalculate centroids as the mean of each cluster
        new_centroids = np.zeros_like(iter_centroids_in)
        for j in range(k):
            members = X[assignments == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)
            else:
                new_centroids[j] = iter_centroids_in[j]

        # Step E. Build table rows to be displayed in the Flask page
        rows = []
        for i in range(len(X)):
            rows.append({
                "id": data[i]["id"],
                "name": data[i]["name"],
                "critic_score": float(X[i, 0]),
                "global_sales": float(X[i, 1]),
                "d1": round(float(distances[i, 0]), 3),
                "d2": round(float(distances[i, 1]), 3),
                "d3": round(float(distances[i, 2]), 3),
                "cluster": int(assignments[i]) + 1,
            })

        counts = [int(np.sum(assignments == j)) for j in range(k)]

        iterations.append({
            "iteration": it + 1,
            "centroids_in": [
                {
                    "cluster": j + 1,
                    "critic_score": round(float(iter_centroids_in[j, 0]), 3),
                    "global_sales": round(float(iter_centroids_in[j, 1]), 3),
                }
                for j in range(k)
            ],
            "rows": rows,
            "counts": counts,
            "wcss": round(wcss, 3),
            "new_centroids": [
                {
                    "cluster": j + 1,
                    "critic_score": round(float(new_centroids[j, 0]), 3),
                    "global_sales": round(float(new_centroids[j, 1]), 3),
                }
                for j in range(k)
            ],
        })

        centroids = new_centroids

    _plot_manual_variance([it["wcss"] for it in iterations])

    return {
        "features": MANUAL_FEATURES,
        "dataset_size": len(data),
        "initial_centroids": INITIAL_CENTROIDS,
        "iterations": iterations,
    }


def _plot_manual_variance(wcss_values):
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(range(1, len(wcss_values) + 1))
    ax.plot(xs, wcss_values, marker="o", color="#4f6ef7",
            linewidth=2.5, markersize=10)
    for x, y in zip(xs, wcss_values):
        ax.annotate(f"{y:.1f}", (x, y),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10, fontweight="bold",
                    color="#1e293b")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=11)
    ax.set_title("Variance Reduction Across K-Means Iterations",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_STATIC_DIR, "kmeans_variance.png"), dpi=150)
    plt.close(fig)


# ===========================================================================
# PART 2  ---  Scikit-learn K-Means Application (1000+ records)
# Features: Critic_Score (0-100), User_Score (0-10)
# Professor's pattern: StandardScaler + KMeans(random_state=42, n_init=10)
# ===========================================================================

APP_FEATURES = ["Critic_Score", "User_Score"]
N_CLUSTERS = 3


def getDataset():
    """Return the full dataset used by the Flask clustering application."""
    return _df[["Name"] + APP_FEATURES].copy().reset_index(drop=True)


def ApplyClusterIngKmeans():
    data = getDataset()

    x = [[row["Critic_Score"], row["User_Score"]] for _, row in data.iterrows()]

    scaler = StandardScaler()
    XScaled = scaler.fit_transform(x)
    model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = model.fit_predict(XScaled)

    # ---- Cluster assignment table (first 30 games as sample) ----
    results = []
    for i in range(min(30, len(data))):
        results.append({
            "name": str(data.loc[i, "Name"])[:60],
            "critic_score": float(data.loc[i, "Critic_Score"]),
            "user_score": float(data.loc[i, "User_Score"]),
            "cluster": int(labels[i]) + 1,
        })

    # ---- Summary of clusters ----
    summary = []
    for j in range(N_CLUSTERS):
        mask = labels == j
        summary.append({
            "cluster": j + 1,
            "count": int(np.sum(mask)),
            "avg_critic": round(float(data.loc[mask, "Critic_Score"].mean()), 2),
            "avg_user": round(float(data.loc[mask, "User_Score"].mean()), 2),
            "min_critic": round(float(data.loc[mask, "Critic_Score"].min()), 2),
            "max_critic": round(float(data.loc[mask, "Critic_Score"].max()), 2),
            "min_user": round(float(data.loc[mask, "User_Score"].min()), 2),
            "max_user": round(float(data.loc[mask, "User_Score"].max()), 2),
        })

    # ---- Centroids (inverse-transformed back to original scale) ----
    centers_original = scaler.inverse_transform(model.cluster_centers_)
    centers = []
    for j in range(N_CLUSTERS):
        centers.append({
            "cluster": j + 1,
            "critic_score": round(float(centers_original[j, 0]), 2),
            "user_score": round(float(centers_original[j, 1]), 2),
        })

    # ---- Scatter plot with clusters and centroids ----
    X_raw = np.array(x)
    _plot_cluster_scatter(X_raw, labels, centers_original)

    return {
        "dataset_size": len(data),
        "features": APP_FEATURES,
        "n_clusters": N_CLUSTERS,
        "results": results,
        "summary": summary,
        "centers": centers,
        "inertia": round(float(model.inertia_), 3),
    }


def _plot_cluster_scatter(X, labels, centers):
    fig, ax = plt.subplots(figsize=(10, 6))
    for j in range(N_CLUSTERS):
        mask = labels == j
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=_CLUSTER_COLORS[j], alpha=0.45, s=22,
                   label=f"Cluster {j+1}")
    ax.scatter(centers[:, 0], centers[:, 1],
               c="black", marker="X", s=300, linewidths=2,
               edgecolors="white", label="Centroids", zorder=5)
    ax.set_xlabel("Critic Score (0-100)", fontsize=11)
    ax.set_ylabel("User Score (0-10)", fontsize=11)
    ax.set_title("K-Means Clustering — Video Games by Critical and User Reception",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_STATIC_DIR, "kmeans_scatter.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pre-compute results at import time so the Flask app stays snappy
# ---------------------------------------------------------------------------
manual_result = runManualKMeans()
app_result = ApplyClusterIngKmeans()


if __name__ == "__main__":
    print("=== Manual K-Means Simulation ===")
    print(f"Dataset size: {manual_result['dataset_size']} records")
    print(f"Features    : {manual_result['features']}")
    for it in manual_result["iterations"]:
        print(f"  Iter {it['iteration']}: WCSS={it['wcss']}  counts={it['counts']}")
        print(f"    New centroids: {it['new_centroids']}")

    print("\n=== Scikit-learn K-Means Application ===")
    print(f"Dataset size: {app_result['dataset_size']} records")
    print(f"Inertia (WCSS): {app_result['inertia']}")
    for s in app_result["summary"]:
        print(f"  Cluster {s['cluster']}: n={s['count']} "
              f"avg_critic={s['avg_critic']} avg_user={s['avg_user']}")
    print(f"Centers: {app_result['centers']}")
