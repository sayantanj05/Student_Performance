import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def correlation_with_target(df, target_col, top_n=10):
    numeric_df = df.select_dtypes(include=np.number)
    if target_col not in numeric_df.columns:
        print(f"Target column '{target_col}' not found in numeric columns.")
        return None
    correlations = numeric_df.corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    print(f"\n========== Correlation with {target_col} ==========")
    print(correlations.head(top_n))
    return correlations


def plot_feature_importance(df, target_col, top_n=15):
    numeric_df = df.select_dtypes(include=np.number)
    if target_col not in numeric_df.columns:
        print(f"Target column '{target_col}' not found.")
        return None
    correlations = numeric_df.corr()[target_col].drop(target_col)
    correlations = correlations.sort_values(key=abs, ascending=True)

    plt.figure(figsize=(10, max(6, len(correlations) * 0.4)))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in correlations.values]
    plt.barh(correlations.index, correlations.values, color=colors)
    plt.title(f"Feature Correlation with {target_col}")
    plt.xlabel("Correlation")
    plt.tight_layout()
    plt.show()


def select_features(df, target_col, correlation_threshold=0.1, exclude=None, include=None):
    if exclude is None:
        exclude = ["Medu", "Fedu", "Dalc", "Walc"]
    if include is None:
        include = ["health"]
    numeric_df = df.select_dtypes(include=np.number)
    correlations = numeric_df.corr()[target_col].drop(target_col)
    important_features = correlations[abs(correlations) >= correlation_threshold]
    if exclude:
        important_features = important_features.drop([c for c in exclude if c in important_features.index], errors="ignore")
    for col in include:
        if col in correlations.index and col not in important_features.index:
            important_features[col] = correlations[col]
    print(f"\n========== Selected Features (|corr| >= {correlation_threshold}) ==========")
    print(important_features.sort_values(key=abs, ascending=False))
    return important_features.sort_values(key=abs, ascending=False).index.tolist()


def feature_selection(df, target_col="G3"):
    print("========== FEATURE SELECTION ==========")
    print("Shape:", df.shape)

    correlations = correlation_with_target(df, target_col)
    if correlations is not None:
        plot_feature_importance(df, target_col)
        selected = select_features(df, target_col)
        print(f"\nSelected features: {selected}")
        return selected
    return []


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_selection.py <path_to_csv> [target_column]")
        sys.exit(1)
    file_path = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else "G3"
    df = pd.read_csv(file_path)
    feature_selection(df, target_col=target)