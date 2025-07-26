import mlflow
import mlflow.sklearn # Specific MLflow module for scikit-learn models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np # For calculating RMSE
import logging
import os

# Import your data loading function from src.data
from src.data import load_and_prepare_data, SCALER_PATH

# Configure logging for better visibility in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- IMPORTANT: Configure MLflow Tracking URI to point to the Dockerized server ---
# The container name 'mlflow-tracking-server' acts as its hostname on the Docker network.
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
logger.info(f"MLFLOW_TRACKING_URI set to: {os.environ['MLFLOW_TRACKING_URI']}")

# This will create the experiment if it doesn't exist, and then activate it.
mlflow.set_experiment("California Housing Training")

def train_and_log_model(model_name, X_train, X_test, y_train, y_test, params):
    """
    Trains a specified regression model, evaluates its performance,
    and logs parameters, metrics, and the model itself to MLflow.
    """
    # Start an MLflow run. Each run captures a single experiment.
    with mlflow.start_run(run_name=f"{model_name}_California_Housing_Run"):
        logger.info(f"Starting MLflow run for {model_name}...")

        # Log model parameters
        mlflow.log_params(params)
        logger.info(f"Logged parameters: {params}")

        # Initialize and train the model based on model_name
        model = None
        if model_name == "LinearRegression":
            model = LinearRegression(**params)
        elif model_name == "RandomForestRegressor":
            model = RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate evaluation metrics for regression
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse) # Root Mean Squared Error
        r2 = r2_score(y_test, predictions) # R-squared score

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2
        }

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        logger.info(f"Logged metrics: {metrics}")

        # Log the trained model to MLflow.
        # 'model' is the artifact path within the run, 'registered_model_name' registers it globally.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model", # Name of the artifact folder within the run
            registered_model_name="CaliforniaHousingRegressor", # Name in MLflow Model Registry
            signature=mlflow.models.infer_signature(X_test, predictions) # Infer input/output schema
        )
        logger.info(f"Model '{model_name}' logged and registered as 'CaliforniaHousingRegressor'.")

        # Log the scaler as an artifact associated with this run
        # This is useful for tracking which scaler was used with which model version
        mlflow.log_artifact(SCALER_PATH, "scaler")
        logger.info(f"Scaler from {SCALER_PATH} logged as an artifact.")

        logger.info(f"MLflow Run finished for {model_name}.")
        return rmse # Return RMSE for easy comparison if needed outside MLflow UI

if __name__ == "__main__":
    logger.info("--- Starting Model Training Script ---")

    # Load and preprocess data using your data.py module
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    logger.info("Data loaded and preprocessed successfully.")

    # --- Train Linear Regression Model ---
    linear_reg_params = {"fit_intercept": True} # Example parameter for Linear Regression
    logger.info("Training Linear Regression model...")
    lr_rmse = train_and_log_model(
        "LinearRegression", X_train, X_test, y_train, y_test, linear_reg_params
    )
    logger.info(f"Linear Regression RMSE: {lr_rmse:.4f}")

    # --- Train Random Forest Regressor Model ---
    random_forest_params = {
        "n_estimators": 100, # Number of trees in the forest
        "max_depth": 10,     # Maximum depth of the tree
        "random_state": 42   # For reproducibility
    }
    logger.info("Training Random Forest Regressor model...")
    rf_rmse = train_and_log_model(
        "RandomForestRegressor", X_train, X_test, y_train, y_test, random_forest_params
    )
    logger.info(f"Random Forest Regressor RMSE: {rf_rmse:.4f}")

    logger.info("\n--- Training Complete ---")
    logger.info("To view MLflow UI, run 'mlflow ui' in your terminal and navigate to http://localhost:5000")
    logger.info("You can compare runs and register the best model from the UI.")