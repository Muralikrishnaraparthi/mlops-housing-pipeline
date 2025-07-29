import os

import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration Paths ---
RAW_DATA_PATH = "data/raw/california_housing.csv"
PROCESSED_DATA_DIR = "data/processed"
SCALER_FILENAME = "scaler.pkl"
SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, SCALER_FILENAME)


def load_raw_data_and_save():
    """
    Loads the California Housing dataset and saves it as CSV.
    Ensures the target column is named 'target'.
    """
    if not os.path.exists(RAW_DATA_PATH):
        print("Downloading California Housing dataset...")
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame.rename(columns={"MedHouseVal": "target"})

        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"Raw data saved to {RAW_DATA_PATH}")
    else:
        print(
        f"Raw data already exists at {RAW_DATA_PATH}. "
        "Loading from file."
    )

    df = pd.read_csv(RAW_DATA_PATH)
    if "MedHouseVal" in df.columns:
        df = df.rename(columns={"MedHouseVal": "target"})

    return df


def preprocess_data(df):
    """
    Preprocesses the housing data:
    - Splits features (X) and target (y)
    - Splits into train/test
    - Scales numerical features
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X.columns, index=X_test.index
    )

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Fitted StandardScaler saved to {SCALER_PATH}")

    return X_train_scaled, X_test_scaled, y_train, y_test


def load_and_prepare_data():
    df = load_raw_data_and_save()
    return preprocess_data(df.copy())


if __name__ == "__main__":
    print("\n--- Running Data Processing Script ---")
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    print("\nData Loading and Preprocessing Complete:")
    print(f"  Raw data file: {RAW_DATA_PATH}")
    print(f"  Fitted scaler file: {SCALER_PATH}")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  X_train (first 5 rows):\n{X_train.head()}")
    print(f"  y_train (first 5 rows):\n{y_train.head()}")
