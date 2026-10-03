import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_model_comparison(summary_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = ["MAE", "MSE", "RMSE", "R2"]
    x = np.arange(len(summary_df))
    width = 0.2

    for i, metric in enumerate(metrics):
        if metric == "R2":
            axes[0].bar(x + i * width, summary_df[metric], width, label=metric)
        else:
            axes[0].bar(x + i * width, summary_df[metric], width, label=metric)

    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Error (lower is better)")
    axes[0].set_title("Model Comparison — Error Metrics")
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(summary_df["Model"], rotation=15)
    axes[0].legend()

    axes[1].bar(summary_df["Model"], summary_df["R2"], color="steelblue")
    axes[1].set_ylabel("R² Score")
    axes[1].set_title("Model Comparison — R² Score (higher is better)")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(summary_df["R2"]):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha="center")

    plt.tight_layout()
    plt.show()


def plot_residuals(results, target_col="G3"):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for i, res in enumerate(results):
        residuals = res["y_test"] - res["y_pred"]
        axes[i].scatter(res["y_pred"], residuals, alpha=0.5, color="coral")
        axes[i].axhline(y=0, color="red", linestyle="--")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Residuals")
        axes[i].set_title(f"{res['Model']} — Residuals")
    plt.suptitle(f"Residual Plots ({target_col})", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_feature_importance_rf(model, feature_names, model_name="Random Forest"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"{model_name} — Feature Importance")
        plt.bar(range(len(importances)), importances[indices], color="steelblue", align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()
    else:
        print(f"{model_name} does not have feature_importances_ attribute.")


def plot_correlation_heatmap(df, title="Feature Correlation Heatmap", save_path=None):
    plt.figure(figsize=(14, 10))
    corr = df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.show()


def plot_target_vs_features(df, target_col="G3", top_n=6):
    numeric_df = df.select_dtypes(include=np.number)
    correlations = numeric_df.corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    top_features = correlations.head(top_n).index.tolist()

    n = len(top_features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(top_features):
        sns.scatterplot(x=df[col], y=df[target_col], ax=axes[i], alpha=0.5, color="steelblue")
        axes[i].set_title(f"{col} vs {target_col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target_col)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Top {top_n} Features vs {target_col}", fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize(results, summary_df, df, target_col="G3", rf_model=None, feature_names=None):
    print("========== VISUALIZATION ==========")

    plot_model_comparison(summary_df)

    plot_residuals(results, target_col=target_col)

    if rf_model is not None and feature_names is not None:
        plot_feature_importance_rf(rf_model, feature_names)

    plot_target_vs_features(df, target_col=target_col)

    corr_path = "correlation_heatmap.png"
    plot_correlation_heatmap(df, save_path=corr_path)

    print("Visualization complete.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualization.py <path_to_csv> [target_column]")
        sys.exit(1)
    file_path = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else "G3"
    df = pd.read_csv(file_path)
    print("Run visualization.py from main.py after training models.")
    print("This module is designed to be imported by main.py.")
    plot_target_vs_features(df, target_col=target)