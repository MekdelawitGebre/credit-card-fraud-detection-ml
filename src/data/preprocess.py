import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

def preprocess_data(csv_path="data/raw/creditcard.csv"):
    # Load data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")

    # Data cleaning
    print("Missing values:", df.isnull().sum().sum())
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Dropped {duplicates} duplicates")

    # EDA summary
    print(df['Class'].value_counts())
    print(df.describe())

    # Feature/target split
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE for imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print("After SMOTE:", y_train_res.value_counts())

    return X_train_res, X_test_scaled, y_train_res, y_test, scaler

if __name__ == "__main__":
    preprocess_data()
