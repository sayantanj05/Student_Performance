import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def remove_missing_values(df):
    before = df.shape[0]
    df = df.dropna()
    after = df.shape[0]
    print(f"Missing values removed: {before - after}")
    return df


def remove_duplicates(df):
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Duplicate rows removed: {before - after}")
    return df


def encode_categorical(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    return df


def scale_features(df, target_col=None):
    scaler = StandardScaler()
    if target_col:
        feature_cols = [c for c in df.columns if c != target_col]
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df, scaler


def preprocess(file_path):
    df = load_data(file_path)
    print("Data loaded successfully.")
    print("Shape before preprocessing:", df.shape)

    df = remove_missing_values(df)
    df = remove_duplicates(df)
    df = encode_categorical(df)

    print("Shape after preprocessing:", df.shape)
    print("Columns:", df.columns.tolist())

    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <path_to_csv>")
        sys.exit(1)
    df = preprocess(sys.argv[1])
    print(df.head())