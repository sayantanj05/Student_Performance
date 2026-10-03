import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_correlation_heatmap(df, title="Correlation Heatmap", save_path=None):
    plt.figure(figsize=(14, 10))
    corr = df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.show()


def plot_target_distribution(df, target_col, title="Target Distribution"):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target_col], kde=True, bins=20, color="skyblue")
    plt.title(title)
    plt.xlabel(target_col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(df, feature_cols, title="Feature Distributions"):
    n = len(feature_cols)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        sns.histplot(df[col], kde=True, bins=20, ax=axes[i], color="salmon")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_pairplot(df, feature_cols, title="Pairplot"):
    sns.pairplot(df[feature_cols])
    plt.suptitle(title, y=1.02)
    plt.show()


def plot_bar_categorical(df, cat_cols, target_col=None, title="Categorical Features vs Target"):
    n = min(len(cat_cols), 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols[:n]):
        if target_col:
            sns.boxplot(x=col, y=target_col, data=df, ax=axes[i])
        else:
            df[col].value_counts().plot(kind="bar", ax=axes[i])
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def eda(df, target_col="G3"):
    print("========== EXPLORATORY DATA ANALYSIS ==========")
    print("Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nStatistical Summary:")
    print(df.describe())

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    print("\nNumerical Columns:", num_cols)
    print("Categorical Columns:", cat_cols)

    if target_col in df.columns:
        plot_target_distribution(df, target_col)

    if num_cols:
        plot_correlation_heatmap(df)
        feature_cols = [c for c in num_cols if c != target_col]
        if feature_cols:
            plot_feature_distributions(df, feature_cols)

    if cat_cols:
        plot_bar_categorical(df, cat_cols, target_col if target_col in df.columns else None)

    print("EDA complete.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python eda.py <path_to_csv> [target_column]")
        sys.exit(1)
    file_path = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else "G3"
    df = pd.read_csv(file_path)
    eda(df, target_col=target)