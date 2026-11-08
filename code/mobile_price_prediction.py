"""
Mobile Phone Price Estimation from Technical Specifications (Regression)
CLA-3 Case Study - Machine Learning

Run from anywhere:
    python code/mobile_price_prediction.py
All paths are taken relative to the project folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # so the graph is saved even without a display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

# ---------------------------------------------------------------- paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "ndtv_data_final.csv")   # Kaggle file
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def load_data(path):
    """Read the Kaggle csv and give the columns short, simple names."""
    data = pd.read_csv(path)
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")      # extra index column in the file
    data = data.rename(columns={
        "Name": "name", "Brand": "brand", "Model": "model",
        "Battery capacity (mAh)": "battery_capacity", "Screen size (inches)": "screen_size",
        "Touchscreen": "touch_screen", "Resolution x": "resolution_x", "Resolution y": "resolution_y",
        "Processor": "processor", "RAM (MB)": "ram", "Internal storage (GB)": "internal_storage",
        "Rear camera": "rear_camera", "Front camera": "front_camera",
        "Operating system": "operating_system", "Wi-Fi": "wi_fi", "Bluetooth": "bluetooth",
        "GPS": "gps", "Number of SIMs": "number_of_sims", "3G": "3g", "4G/ LTE": "4g_lte",
        "Price": "price"})
    return data


# ---------------------------------------------------------------- 1. load data
df = load_data(DATA_PATH)
print("Dataset shape (rows, columns):", df.shape)
print("\nColumns and data types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum().to_string())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nTarget column = 'price' (Indian Rupees)")
print(df["price"].describe().round(2))

# ---------------------------------------------------------------- 2. preprocessing
# 'name' and 'model' are just identifiers (almost unique for every phone), so they are dropped.
df = df.drop(columns=["name", "model"])

# Yes/No columns -> 1/0
yes_no_cols = ["touch_screen", "wi_fi", "bluetooth", "gps", "3g", "4g_lte"]
for col in yes_no_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# any missing values are filled (median for numbers). The dataset has none, but this keeps code safe.
num_cols_all = df.select_dtypes(include=np.number).columns
df[num_cols_all] = df[num_cols_all].fillna(df[num_cols_all].median())

X = df.drop(columns=["price"])
y = df["price"]

cat_cols = ["brand", "operating_system"]            # converted with one-hot encoding
num_cols = [c for c in X.columns if c not in cat_cols]

# scaling for numbers, one-hot for brand/OS
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# ---------------------------------------------------------------- 3. train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))

# ---------------------------------------------------------------- 4. models
models = {
    "Baseline (mean price)": DummyRegressor(strategy="mean"),   # always predicts the training mean
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

rows = []
fitted = {}
preds = {}
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, reg in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", reg)])
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, p)
    rmse = np.sqrt(mean_squared_error(y_test, p))
    r2 = r2_score(y_test, p)
    cv_r2 = cross_val_score(pipe, X, y, cv=kf, scoring="r2").mean()   # extra check on all data
    rows.append({"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
                 "R2": round(r2, 4), "CV_R2_5fold": round(cv_r2, 4)})
    fitted[name] = pipe
    preds[name] = p
    print(f"{name:18s} MAE={mae:10.2f}  RMSE={rmse:10.2f}  R2={r2:.4f}  (5-fold CV R2={cv_r2:.4f})")

results = pd.DataFrame(rows)
results.to_csv(os.path.join(RESULT_DIR, "model_comparison.csv"), index=False)
print("\nModel comparison:")
print(results.to_string(index=False))

# ---------------------------------------------------------------- 5. best model (highest R2 on the test set)
best_name = results[results["Model"] != "Baseline (mean price)"].sort_values("R2", ascending=False).iloc[0]["Model"]
best_pred = preds[best_name]
print("\nBest model on the test set:", best_name)

# ---------------------------------------------------------------- 6. actual vs predicted table
test_info = load_data(DATA_PATH).loc[X_test.index, ["name", "brand"]]
out = pd.DataFrame({
    "name": test_info["name"].values,
    "actual_price": y_test.values,
    "predicted_price": best_pred.round(0),
})
out["abs_error"] = (out["actual_price"] - out["predicted_price"]).abs()
out["error_percent"] = (100 * out["abs_error"] / out["actual_price"]).round(1)
out.to_csv(os.path.join(RESULT_DIR, "test_predictions_best_model.csv"), index=False)

print("\nFirst 10 test phones (actual vs predicted, best model):")
print(out.head(10).to_string(index=False))

# one close prediction and one bad prediction
# close example: smallest % error among phones with price above the median (so it is a meaningful phone)
mid = out[out["actual_price"] >= out["actual_price"].median()]
good = mid.sort_values("error_percent").iloc[0]
bad = out.sort_values("abs_error", ascending=False).iloc[0]
examples = pd.DataFrame([good, bad], index=["close prediction", "wrong prediction"])
examples.to_csv(os.path.join(RESULT_DIR, "example_predictions.csv"))
print("\nExamples:")
print(examples.to_string())

# specs of those two phones
raw = load_data(DATA_PATH)
cols_show = ["name", "brand", "battery_capacity", "screen_size", "ram", "internal_storage",
             "rear_camera", "front_camera", "operating_system", "4g_lte", "price"]
print("\nSpecs of the two example phones:")
print(raw[raw["name"].isin([good["name"], bad["name"]])][cols_show].to_string(index=False))

# error summary
print("\nMedian absolute error (best model): %.0f" % out["abs_error"].median())
print("Phones with error under 20%%: %.1f%%" % (100 * (out["error_percent"] < 20).mean()))

# ---------------------------------------------------------------- 7. feature importance (Random Forest)
rf_fit = fitted["Random Forest"]
prep = rf_fit.named_steps["prep"]
names = list(num_cols) + list(prep.named_transformers_["cat"].get_feature_names_out(cat_cols))
imp = pd.Series(rf_fit.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
imp.head(10).to_csv(os.path.join(RESULT_DIR, "top_features.csv"), header=["importance"])
print("\nTop 8 important features (Random Forest):")
print(imp.head(8).round(4).to_string())
print("Total importance of all brand columns: %.4f" % imp[[n for n in names if n.startswith("brand_")]].sum())
print("Total importance of all OS columns: %.4f" % imp[[n for n in names if n.startswith("operating_system_")]].sum())

# ---------------------------------------------------------------- 8. worked example for Random Forest
# show how the forest averages individual trees for the 'close prediction' phone
rf_pipe = fitted["Random Forest"]
x_row = X_test.loc[[X_test.index[out.index[out["name"] == good["name"]][0]]]]
x_row_t = rf_pipe.named_steps["prep"].transform(x_row)
trees = rf_pipe.named_steps["model"].estimators_
tree_preds = [t.predict(x_row_t)[0] for t in trees]
print("\nRandom Forest worked example for:", good["name"])
print("Number of trees:", len(trees))
print("First 5 tree predictions:", [round(v) for v in tree_preds[:5]])
print("Average of all trees: %.0f" % np.mean(tree_preds))


# ---------------------------------------------------------------- 8b. tiny linear regression example (2 features only)
# Simple model with only RAM (GB) and storage (GB), used just to show how y = b0 + b1*x1 + b2*x2 works.
small_train = load_data(DATA_PATH).loc[X_train.index]
small_X = pd.DataFrame({"ram_gb": small_train["ram"] / 1000, "storage_gb": small_train["internal_storage"]})
small_lr = LinearRegression().fit(small_X, small_train["price"])
b0, (b1, b2) = small_lr.intercept_, small_lr.coef_
print("\nSimple 2-feature Linear Regression (RAM in GB, storage in GB):")
print("b0 = %.2f, b1 (RAM) = %.2f, b2 (storage) = %.2f" % (b0, b1, b2))
ram_gb, storage_gb = 3.0, 32.0   # LG W30 specs
print("Prediction for RAM=3 GB, storage=32 GB: %.2f" % (b0 + b1 * ram_gb + b2 * storage_gb))

# ---------------------------------------------------------------- 9. graph: actual vs predicted
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, name in zip(axes, ["Linear Regression", "Random Forest"]):
    ax.scatter(y_test, preds[name], s=14, alpha=0.6)
    lim = max(y_test.max(), preds[name].max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", label="perfect prediction")
    ax.set_xlabel("Actual price (Rs.)")
    ax.set_ylabel("Predicted price (Rs.)")
    r2v = r2_score(y_test, preds[name])
    ax.set_title(f"{name} (R2 = {r2v:.3f})")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "actual_vs_predicted.png"), dpi=200)
print("\nSaved: results/model_comparison.csv, results/actual_vs_predicted.png, "
      "results/test_predictions_best_model.csv, results/example_predictions.csv")
