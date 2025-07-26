import mlflow
import mlflow.sklearn # Specific MLflow module for scikit-learn models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import logging
import os

# Import your data loading function from src.data
from src.data import load_and_prepare_data, SCALER_PATH

# Configure logging for better visibility in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- IMPORTANT: Configure MLflow Tracking URI to point to the Dockerized server ---
os.environ["MLFLOW_TRACKING_URI"] = "http://host.docker.internal:5000"
logger.info(f"MLFLOW_TRACKING_URI set to: {os.environ['MLFLOW_TRACKING_URI']}")

mlflow.set_experiment("California Housing Training")

def train_and_log_model(model_name, X_train, X_test, y_train, y_test, params):
    """
    Trains a specified regression model, evaluates its performance,
    and logs parameters, metrics, and the model itself to MLflow.
    """
    with mlflow.start_run(run_name=f"{model_name}_California_Housing_Run"):
        logger.info(f"Starting MLflow run for {model_name}...")

        mlflow.log_params(params)
        logger.info(f"Logged parameters: {params}")

        if model_name == "LinearRegression":
            model = LinearRegression(**params)
        elif model_name == "RandomForestRegressor":
            model = RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2
        }

        mlflow.log_metrics(metrics)
        logger.info(f"Logged metrics: {metrics}")

        # Fix: Use X_test (scaled) for signature
        from mlflow.models import infer_signature
        signature = infer_signature(X_test, predictions)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="CaliforniaHousingRegressor",
            signature=signature
        )
        logger.info(f"Model '{model_name}' logged and registered as 'CaliforniaHousingRegressor'.")

        mlflow.log_artifact(SCALER_PATH, "scaler")
        logger.info(f"Scaler from {SCALER_PATH} logged as an artifact.")

        logger.info(f"MLflow Run finished for {model_name}.")
        return rmse

if __name__ == "__main__":
    logger.info("--- Starting Model Training Script ---")

    X_train, X_test, y_train, y_test = load_and_prepare_data()
    logger.info("Data loaded and preprocessed successfully.")

    linear_reg_params = {"fit_intercept": True}
    logger.info("Training Linear Regression model...")
    lr_rmse = train_and_log_model(
        "LinearRegression", X_train, X_test, y_train, y_test, linear_reg_params
    )
    logger.info(f"Linear Regression RMSE: {lr_rmse:.4f}")

    random_forest_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    logger.info("Training Random Forest Regressor model...")
    rf_rmse = train_and_log_model(
        "RandomForestRegressor", X_train, X_test, y_train, y_test, random_forest_params
    )
    logger.info(f"Random Forest Regressor RMSE: {rf_rmse:.4f}")

    logger.info("\n--- Training Complete ---")
    logger.info("To view MLflow UI, run 'mlflow ui' and visit http://localhost:5000")
    logger.info("You can compare runs and promote the best model from the UI.")
