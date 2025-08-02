import os
import shutil
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
NEW_DATA_DIR = "data/new_data"
ARCHIVE_DIR = "data/archive"

EXPECTED_COLS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude", "target"
]


def load_raw_data_and_save():
    """
    Loads the California Housing dataset and saves it as CSV.
    If new files exist in data/new_data/, they are appended with validation.
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

    # --- Append validated new data files ---
    if os.path.exists(NEW_DATA_DIR):
        new_files = [f for f in os.listdir(NEW_DATA_DIR) if f.endswith(".csv")]
        if new_files:
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
            for f in new_files:
                fpath = os.path.join(NEW_DATA_DIR, f)
                try:
                    new_df = pd.read_csv(fpath)
                    if (
                        "target" not in new_df.columns
                        and "MedHouseVal" in new_df.columns
                    ):
                        new_df.rename(
                            columns={"MedHouseVal": "target"}, inplace=True
                        )

                    if sorted(new_df.columns) == sorted(EXPECTED_COLS):
                        df = pd.concat([df, new_df], ignore_index=True)
                        print(f"✅ Appended valid data from: {f}")
                    else:
                        print(f"❌ Skipped file due to column mismatch: {f}")
                        print(f"   Expected: {EXPECTED_COLS}")
                        print(f"   Found:    {list(new_df.columns)}")
                except Exception as e:
                    print(f"❌ Error reading {f}: {e}")

                archive_path = os.path.join(ARCHIVE_DIR, f)
                shutil.move(fpath, archive_path)
        else:
            print("No new data files found in data/new_data/")
    else:
        print("No data/new_data/ directory found.")

    return df


def preprocess_data(df):
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
