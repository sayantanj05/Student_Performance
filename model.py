import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


def prepare_data(df, target_col="G3", feature_cols=None):
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    df_model = df[feature_cols + [target_col]].copy()
    le = LabelEncoder()
    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = le.fit_transform(df_model[col])
    X = df_model[feature_cols]
    y = df_model[target_col]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    return models


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'=' * 40}")
    print(f"  {model_name}")
    print(f"{'=' * 40}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

    return {"Model": model_name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "y_pred": y_pred, "y_test": y_test}


def plot_predictions(results, target_col="G3"):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for i, res in enumerate(results):
        axes[i].scatter(res["y_test"], res["y_pred"], alpha=0.5, color="steelblue")
        min_val = min(res["y_test"].min(), res["y_pred"].min())
        max_val = max(res["y_test"].max(), res["y_pred"].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
        axes[i].set_xlabel("Actual")
        axes[i].set_ylabel("Predicted")
        axes[i].set_title(res["Model"])
        axes[i].legend()
    plt.suptitle(f"Actual vs Predicted ({target_col})", fontsize=14)
    plt.tight_layout()
    plt.show()


def train_and_evaluate(df, target_col="G3", feature_cols=None):
    print("========== MODEL TRAINING & EVALUATION ==========")
    X, y = prepare_data(df, target_col=target_col, feature_cols=feature_cols)
    print(f"Features used: {feature_cols}")
    print(f"Target: {target_col}")

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = get_models()
    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        res = evaluate_model(model, X_test, y_test, name)
        results.append(res)

    summary_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("y_pred", "y_test")} for r in results])
    print("\n========== MODEL COMPARISON ==========")
    print(summary_df.to_string(index=False))

    best_model = summary_df.loc[summary_df["R2"].idxmax()]
    print(f"\nBest Model: {best_model['Model']} (R² = {best_model['R2']:.4f})")

    plot_predictions(results, target_col=target_col)

    return results, summary_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python model.py <path_to_csv> [target_column]")
        sys.exit(1)
    file_path = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else "G3"
    df = pd.read_csv(file_path)
    train_and_evaluate(df, target_col=target)