import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def predict_g4(df, target_col="G3"):
    print("========== PREDICT G4 FOR ALL STUDENTS ==========")

    feature_cols = [c for c in df.columns if c not in ("G3", "G4")]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    df["Student_ID"] = range(1, len(df) + 1)

    le = None
    for col in X.select_dtypes(include='object').columns:
        le = __import__('sklearn.preprocessing', fromlist=['LabelEncoder']).LabelEncoder()
        X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f"Model Accuracy on Test Set:")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²:  {r2:.4f}")

    X_all = X.copy()
    df["G4_Predicted"] = model.predict(X_all)

    print(f"\n========== PREDICTED RESULTS (All Students) ==========")
    result_df = df[["Student_ID", "school", "sex", "age", "G1", "G2", "G3", "G4_Predicted"]].copy()
    result_df["G4_Predicted"] = result_df["G4_Predicted"].round(2)
    print(result_df.to_string(index=False))

    print(f"\n========== STATS ON PREDICTED G4 ==========")
    print(f"Mean G4:   {df['G4_Predicted'].mean():.2f}")
    print(f"Min G4:    {df['G4_Predicted'].min():.2f}")
    print(f"Max G4:    {df['G4_Predicted'].max():.2f}")
    print(f"Std G4:    {df['G4_Predicted'].std():.2f}")

    plt.figure(figsize=(10, 5))
    plt.scatter(df["G3"], df["G4_Predicted"], alpha=0.5, color="steelblue")
    min_val = min(df["G3"].min(), df["G4_Predicted"].min())
    max_val = max(df["G3"].max(), df["G4_Predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
    plt.xlabel("G3 (Actual)")
    plt.ylabel("G4 (Predicted)")
    plt.title("G3 vs Predicted G4")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    df["G4_Predicted"].hist(bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Predicted G4")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted G4")
    plt.tight_layout()
    plt.show()

    result_df.to_csv("predicted_G4_results.csv", index=False)
    print("\nSaved predictions to: predicted_G4_results.csv")

    return df, result_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python prediction.py <path_to_csv>")
        sys.exit(1)
    file_path = sys.argv[1]
    df = pd.read_csv(file_path)
    predict_g4(df)