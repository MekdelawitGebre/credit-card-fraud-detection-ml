# src/preprocessing/preprocess.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def create_dir(path):
    """Create directory if it does not exist"""
    os.makedirs(path, exist_ok=True)
    return path

# ----------------------------
# 1. Load Dataset
# ----------------------------
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print(f"[INFO] Dataset loaded: {df.shape}")
    return df

# ----------------------------
# 2. Data Cleaning
# ----------------------------
def clean_data(df):
    """Remove duplicates and handle missing values"""
    df = df.drop_duplicates()
    df = df.fillna(df.median())
    print(f"[INFO] Data cleaned: {df.shape}, Missing values: {df.isnull().sum().sum()}")
    return df

# ----------------------------
# 3. Feature Engineering (example)
# ----------------------------
def feature_engineering(df):
    """Add engineered features if necessary (example: log transform of Amount)"""
    df['Amount_log'] = df['Amount'].apply(lambda x: 0 if x == 0 else np.log(x))
    return df

# ----------------------------
# 4. Scaling / Normalization
# ----------------------------
def scale_features(df, columns, save_dir="data/processed"):
    create_dir(save_dir)
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")
    print(f"[INFO] Scaler saved to {save_dir}/scaler.pkl")
    return df

# ----------------------------
# 5. Split Data
# ----------------------------
def split_data(df, target='Class', test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, test sets
    """
    X = df.drop(target, axis=1)
    y = df[target]

    # Train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Train vs validation
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size,
        stratify=y_train_val, random_state=random_state
    )

    print(f"[INFO] Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# ----------------------------
# 6. Handle Class Imbalance
# ----------------------------
def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("[INFO] Class distribution after SMOTE:")
    print(y_res.value_counts())
    return X_res, y_res

# ----------------------------
# 7. Save Processed Data
# ----------------------------
def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, save_dir="data/processed"):
    create_dir(save_dir)
    X_train.to_csv(f"{save_dir}/X_train.csv", index=False)
    y_train.to_csv(f"{save_dir}/y_train.csv", index=False)
    X_val.to_csv(f"{save_dir}/X_val.csv", index=False)
    y_val.to_csv(f"{save_dir}/y_val.csv", index=False)
    X_test.to_csv(f"{save_dir}/X_test.csv", index=False)
    y_test.to_csv(f"{save_dir}/y_test.csv", index=False)
    print(f"[INFO] Processed datasets saved to {save_dir}")

# ----------------------------
# 8. Run Full Pipeline
# ----------------------------
def run_pipeline(raw_file="data/raw/creditcard.csv"):
    df = load_dataset(raw_file)
    df = clean_data(df)
    # Optional: feature engineering if needed
    # df = feature_engineering(df)
    df = scale_features(df, columns=['Time', 'Amount'])
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_res, y_train_res = balance_data(X_train, y_train)
    save_processed_data(X_train_res, X_val, X_test, y_train_res, y_val, y_test)
    print("✅ Full preprocessing pipeline completed successfully.")


