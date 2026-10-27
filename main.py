import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import functions from all modules
from preprocessing import preprocess, load_data, remove_missing_values, remove_duplicates, encode_categorical
from eda import eda
from feature_selection import feature_selection, select_features, correlation_with_target
from model import train_and_evaluate, prepare_data, split_data, get_models, evaluate_model
from prediction import predict_g4
from visualization import visualize, plot_model_comparison, plot_residuals, plot_feature_importance_rf, plot_target_vs_features, plot_correlation_heatmap


def find_dataset():
    """
    Find the dataset CSV file. First check local files, then look in common locations.
    """
    # Check current directory
    for file in os.listdir('.'):
        if file.endswith('.csv') and 'student' in file.lower():
            print(f"Found dataset in current directory: {file}")
            return file
    
    # Check common kagglehub cache location
    cache_dir = os.path.expanduser('~/.cache/kagglehub/datasets/larsen0966/student-performance-data-set')
    if os.path.exists(cache_dir):
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.join(root, file)
                    print(f"Found dataset in cache: {full_path}")
                    return full_path
    
    print("Error: Could not find student dataset CSV file.")
    print("Please provide the path as argument: python main.py <path_to_csv>")
    sys.exit(1)


def main():
    """
    Main function to run all Python files sequentially.
    """
    print("=" * 60)
    print("STUDENT PERFORMANCE DATA ANALYSIS PIPELINE")
    print("=" * 60)
    print()
    
    # Get dataset path
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"Using dataset from command line: {data_path}")
    else:
        data_path = find_dataset()
    
    if not os.path.exists(data_path):
        print(f"Error: File '{data_path}' does not exist.")
        sys.exit(1)
    
    target_col = 'G3'
    print(f"Target column: {target_col}")
    print()
    
    # Step 1: Preprocessing
    print("=" * 60)
    print("STEP 1: PREPROCESSING")
    print("=" * 60)
    df_processed = preprocess(data_path)
    print()
    
    # Step 2: Exploratory Data Analysis
    print("=" * 60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    eda(df_processed, target_col=target_col)
    plt.close('all')  # Close plots to continue
    print()
    
    # Step 3: Feature Selection
    print("=" * 60)
    print("STEP 3: FEATURE SELECTION")
    print("=" * 60)
    selected_features = feature_selection(df_processed, target_col=target_col)
    plt.close('all')  # Close plots to continue
    print()
    
    # Step 4: Model Training and Evaluation
    print("=" * 60)
    print("STEP 4: MODEL TRAINING & EVALUATION")
    print("=" * 60)
    results, summary_df = train_and_evaluate(df_processed, target_col=target_col, feature_cols=selected_features)
    plt.close('all')  # Close plots to continue
    print()
    
    # Get the trained Random Forest model for visualization
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    df_model = df_processed.copy()
    le = LabelEncoder()
    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = le.fit_transform(df_model[col])
    
    X = df_model[selected_features] if selected_features else df_model.drop(columns=[target_col])
    y = df_model[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Step 5: Visualization
    print("=" * 60)
    print("STEP 5: VISUALIZATION")
    print("=" * 60)
    visualize(results, summary_df, df_processed, target_col=target_col, rf_model=rf_model, feature_names=X.columns.tolist())
    plt.close('all')  # Close plots to continue
    print()
    
    # Step 6: Prediction (G4 prediction)
    print("=" * 60)
    print("STEP 6: PREDICTION")
    print("=" * 60)
    # For prediction, use original preprocessed data or reload
    df_pred = load_data(data_path)
    df_pred = remove_missing_values(df_pred)
    df_pred = remove_duplicates(df_pred)
    df_pred = encode_categorical(df_pred)
    pred_df, result_df = predict_g4(df_pred, target_col=target_col)
    plt.close('all')  # Close plots to continue
    print()
    
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Results saved to: predicted_G4_results.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
