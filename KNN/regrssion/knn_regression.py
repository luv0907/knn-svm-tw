import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import re

CSV_PATH = os.path.join("KNN", "regrssion", "house_prices.csv")
MODEL_PATH = os.path.join("KNN", "regrssion", "knn_model.joblib")
PRED_PLOT = os.path.join("KNN", "regrssion", "pred_vs_actual.png")
RESID_PLOT = os.path.join("KNN", "regrssion", "residuals.png")


def detect_target(df):
    # prefer column with 'price' in name, else numeric last column
    price_cols = [c for c in df.columns if "price" in c.lower()]
    if price_cols:
        return price_cols[0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        return numeric_cols[-1]
    return df.columns[-1]


def load_and_clean(path=CSV_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}")
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def parse_amount(s: str):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    num = re.search(r"([0-9]+\.?[0-9]*)", s)
    if not num:
        return np.nan
    val = float(num.group(1))
    if "lac" in s.lower():
        return val * 1e5
    if "cr" in s.lower():
        return val * 1e7
    return val


def parse_sqft(s: str):
    if pd.isna(s):
        return np.nan
    s = str(s).lower()
    num = re.search(r"([0-9]+\.?[0-9]*)", s)
    return float(num.group(1)) if num else np.nan


def parse_floor(s: str):
    if pd.isna(s):
        return (np.nan, np.nan)
    s = str(s)
    nums = re.findall(r"(\d+)", s)
    if not nums:
        return (np.nan, np.nan)
    if len(nums) == 1:
        return (float(nums[0]), np.nan)
    return (float(nums[0]), float(nums[1]))


def extract_bhk(title: str):
    if pd.isna(title):
        return np.nan
    m = re.search(r"(\d+)\s*BHK", str(title), re.IGNORECASE)
    if m:
        return float(m.group(1))
    return np.nan


def parse_parking(s: str):
    if pd.isna(s):
        return 0.0
    m = re.search(r"(\d+)", str(s))
    return float(m.group(1)) if m else 0.0


def preprocess_and_train(df):
    target = detect_target(df)
    X = df.drop(columns=[target])
    y = df[target]

    # Feature engineering
    X = X.copy()
    X["amount_rupees"] = X.get("Amount(in rupees)", "").apply(parse_amount)
    X["carpet_sqft"] = X.get("Carpet Area", "").apply(parse_sqft)
    X["super_area_sqft"] = X.get("Super Area", "").apply(parse_sqft)
    X["plot_area_sqft"] = X.get("Plot Area", "").apply(parse_sqft)
    X["dimensions"] = X.get("Dimensions", np.nan).apply(_safe_float)
    X["bhk"] = X.get("Title", "").apply(extract_bhk)
    X[["floor_num", "total_floors"]] = X.get("Floor", np.nan).apply(lambda s: pd.Series(parse_floor(s)))
    X["parking_count"] = X.get("Car Parking", "").apply(parse_parking)
    for col in ["Bathroom", "Balcony"]:
        if col in X.columns:
            X[col] = X[col].apply(_safe_float)

    drop_cols = [c for c in ["Index", "Title", "Description", "Society", "Amount(in rupees)"] if c in X.columns]
    X = X.drop(columns=drop_cols)

    # Define numeric and categorical columns after engineering
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=[object, "category"]).columns.tolist()

    # Impute numerics with median and categoricals with mode
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

    knn = KNeighborsRegressor()
    pipe = Pipeline([("pre", preprocessor), ("knn", knn)])

    param_grid = {
        "knn__n_neighbors": list(range(3, 21, 2)),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }

    gs = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    y_pred = best.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Plots
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    try:
        m, b = np.polyfit(y_test, y_pred, 1)
        plt.plot(y_test, m * y_test + b, color="red", linewidth=1)
    except Exception:
        pass
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(PRED_PLOT)
    plt.close()

    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30)
    plt.title("Residuals")
    plt.tight_layout()
    plt.savefig(RESID_PLOT)
    plt.close()

    joblib.dump(best, MODEL_PATH)

    results = {
        "best_params": gs.best_params_,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "model_path": MODEL_PATH,
        "pred_plot": PRED_PLOT,
        "resid_plot": RESID_PLOT,
    }
    return results


if __name__ == "__main__":
    print("Loading dataset:", CSV_PATH)
    df = load_and_clean()
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Detecting target and training...")
    res = preprocess_and_train(df)
    print("Best params:", res["best_params"]) 
    print(f"MAE: {res['mae']:.4f}, RMSE: {res['rmse']:.4f}, R2: {res['r2']:.4f}")
    print("Saved model to:", res["model_path"])