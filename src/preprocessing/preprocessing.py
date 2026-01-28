import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

class Preprocessor:
    """
    Professional Preprocessor for Credit Card Fraud Detection.
    Includes: cleaning, scaling, SMOTE, train/test split, saving datasets.
    """

    def __init__(self, raw_path, processed_dir="data/processed", models_dir="models"):
        self.raw_path = raw_path
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self.scaler = None

    def load_data(self):
        """Load dataset from CSV"""
        df = pd.read_csv(self.raw_path)
        print(f"Dataset loaded with shape: {df.shape}")
        return df

    def clean_data(self, df):
        """Remove duplicates and handle missing values"""
        df = df.drop_duplicates()
        missing = df.isnull().sum().sum()
        if missing > 0:
            df = df.fillna(df.median())
        print(f"After cleaning: {df.shape}, missing values: {df.isnull().sum().sum()}")
        return df

    def scale_features(self, X_train, X_test, features=["Time","Amount"], log_transform=True):
        """Scale numeric features"""
        if log_transform and "Amount" in features:
            X_train['Amount'] = np.log1p(X_train['Amount'])
            X_test['Amount'] = np.log1p(X_test['Amount'])
        self.scaler = StandardScaler()
        X_train[features] = self.scaler.fit_transform(X_train[features])
        X_test[features] = self.scaler.transform(X_test[features])
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.pkl"))
        return X_train, X_test

    def balance_classes(self, X_train, y_train):
        """Use SMOTE to handle class imbalance"""
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print("Class distribution after SMOTE:", y_res.value_counts())
        return X_res, y_res

    def train_test_split(self, df, target="Class", test_size=0.2, random_state=42):
        """Split dataset into train/test (stratified)"""
        from sklearn.model_selection import train_test_split
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def save_datasets(self, X_train, X_test, y_train, y_test):
        """Save preprocessed datasets to CSV"""
        X_train.to_csv(os.path.join(self.processed_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(self.processed_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(self.processed_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.processed_dir, "y_test.csv"), index=False)
        print(f"Datasets saved to {self.processed_dir}")

    def run(self):
        """Full preprocessing pipeline"""
        df = self.load_data()
        df = self.clean_data(df)
        X_train, X_test, y_train, y_test = self.train_test_split(df)
        X_train, X_test = self.scale_features(X_train, X_test)
        X_train, y_train = self.balance_classes(X_train, y_train)
        self.save_datasets(X_train, X_test, y_train, y_test)
        print("✅ Full preprocessing pipeline completed.")
