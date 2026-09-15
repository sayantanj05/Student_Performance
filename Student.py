import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

path = kagglehub.dataset_download("larsen0966/student-performance-data-set")

print("Path to dataset files:", path)

csv_file = os.path.join(path, [f for f in os.listdir(path) if f.endswith(".csv")][0])
df = pd.read_csv(csv_file)

print("Dataset loaded successfully ")
print("Dataset downloaded to:")
print(path)
print(df.head())
print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATASET INFO ==========")
print(df.info())
print("\n========== SUMMARY STATISTICS ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

print("\nNumerical Columns")
print(num_cols)

print("\nCategorical Columns")
print(cat_cols)