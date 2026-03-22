import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as compute_r2

# ---------------------------------------------------------------------------
# Paths relative to THIS script's location
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH = os.path.join(_SCRIPT_DIR, "dataset_steam.csv")
_STATIC_DIR = os.path.join(_SCRIPT_DIR, "static")
_PLOT_PATH = os.path.join(_STATIC_DIR, "regression_plot.png")

# ---------------------------------------------------------------------------
# 1. Read dataset (no header)
# ---------------------------------------------------------------------------
df = pd.read_csv(
    _CSV_PATH,
    header=None,
    names=["user_id", "game_name", "action_type", "value", "extra"],
)

# ---------------------------------------------------------------------------
# 2. Games owned per user (count purchase rows)
# ---------------------------------------------------------------------------
games_per_user = (
    df[df["action_type"] == "purchase"]
    .groupby("user_id")
    .size()
    .reset_index(name="games_owned")
)

# ---------------------------------------------------------------------------
# 3. Total play-hours per user (sum value where action == play)
# ---------------------------------------------------------------------------
hours_per_user = (
    df[df["action_type"] == "play"]
    .groupby("user_id")["value"]
    .sum()
    .reset_index(name="total_hours")
)

# ---------------------------------------------------------------------------
# 4. Merge into a single DataFrame
# ---------------------------------------------------------------------------
user_df = pd.merge(games_per_user, hours_per_user, on="user_id", how="inner")

# ---------------------------------------------------------------------------
# 5. Train Linear Regression   X = games_owned  ->  y = total_hours
# ---------------------------------------------------------------------------
X = user_df[["games_owned"]].values
y = user_df["total_hours"].values

model = LinearRegression()
model.fit(X, y)

slope = float(model.coef_[0])
intercept = float(model.intercept_)
r2 = float(compute_r2(y, model.predict(X)))

# ---------------------------------------------------------------------------
# 6. Scatter plot + regression line  ->  static/regression_plot.png
# ---------------------------------------------------------------------------
os.makedirs(_STATIC_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y, alpha=0.3, s=10, label="Users")

x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
ax.plot(x_range, model.predict(x_range), color="red", linewidth=2,
        label=f"y = {slope:.2f}x + {intercept:.2f}  (R² = {r2:.4f})")

ax.set_xlabel("Games Owned")
ax.set_ylabel("Total Hours Played")
ax.set_title("Steam: Games Owned vs Total Hours Played")
ax.legend()
fig.tight_layout()
fig.savefig(_PLOT_PATH, dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Public exports
# ---------------------------------------------------------------------------

def predict_hours(games_owned):
    """Predict total hours played given the number of games owned."""
    return model.predict(np.array([[games_owned]]))[0]


# ---------------------------------------------------------------------------
# 8. Print stats when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Steam Linear Regression ===")
    print(f"Users in dataset : {len(user_df)}")
    print(f"Slope            : {slope:.4f}")
    print(f"Intercept        : {intercept:.4f}")
    print(f"R² Score         : {r2:.4f}")
    print(f"\nExample predictions:")
    for n in [5, 10, 25, 50, 100]:
        print(f"  {n:>3} games -> {predict_hours(n):,.1f} hours")
    print(f"\nPlot saved to: {_PLOT_PATH}")
