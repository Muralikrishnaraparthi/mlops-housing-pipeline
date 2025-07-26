import mlflow
import mlflow.pyfunc
from flask import Flask, request, jsonify, Response
import os
import logging
import pandas as pd
import joblib
from pydantic import BaseModel, Field, ValidationError
from typing import List

# --- Prometheus Client Imports (for Monitoring Bonus) ---
from prometheus_client import generate_latest, Counter, Histogram

# --- Configuration ---
# Basic logging setup: logs to console (stdout/stderr) which Docker captures
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow Model Configuration
MODEL_NAME = "CaliforniaHousingRegressor"
# Using lowercase 'production' for alias, as this has shown better consistency
# (ensure your MLflow UI alias matches exactly: http://localhost:5000 -> Models -> CaliforniaHousingRegressor -> your_version -> Aliases)
MODEL_ALIAS = "production"
MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# IMPORTANT: Configure MLflow Tracking URI for Docker Container-to-Container Communication
# This API container accesses the MLflow server container by its service name 'mlflow-server'.
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow-server:5000"
logger.info(f"MLFLOW_TRACKING_URI set to: {os.environ['MLFLOW_TRACKING_URI']}")

# Scaler Path (relative to the container's working directory /app)
SCALER_CONTAINER_PATH = "data/processed/scaler.pkl"

# Expected features for the model input - MUST match training features order/names
EXPECTED_FEATURES = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

app = Flask(__name__)

# --- Model and Scaler Loading ---
model = None
scaler = None

def load_artifacts():
    """Loads the MLflow model and the DVC-tracked scaler."""
    global model, scaler
    try:
        logger.info("Attempting to load artifacts...")
        logger.info(f"Loading MLflow model from URI: {MLFLOW_MODEL_URI}")
        # This is where the model is downloaded/loaded from the MLflow Tracking Server
        model = mlflow.pyfunc.load_model(model_uri=MLFLOW_MODEL_URI)
        logger.info(f"MLflow model '{MODEL_NAME}' with alias '{MODEL_ALIAS}' loaded successfully.")

        logger.info(f"Loading scaler from path: {SCALER_CONTAINER_PATH}")
        # This is where the DVC-managed scaler.pkl file is loaded from inside the container
        if os.path.exists(SCALER_CONTAINER_PATH):
            scaler = joblib.load(SCALER_CONTAINER_PATH)
            logger.info(f"Scaler loaded successfully.")
        else:
            logger.error(f"Scaler file NOT FOUND at {SCALER_CONTAINER_PATH}. This is critical.")
            scaler = None # Ensure scaler is None if not found

    except Exception as e:
        logger.critical(f"CRITICAL ERROR during artifact loading: {e}", exc_info=True)
        model = None
        scaler = None

# Load artifacts when the Flask app starts (within Flask's application context)
with app.app_context():
    load_artifacts()

# --- Prometheus Metrics (for Monitoring Bonus) ---
# Counter for total prediction requests
PREDICTION_REQUESTS = Counter('prediction_requests_total', 'Total number of prediction requests')
# Histogram for prediction latency
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Latency of prediction requests')


# --- Pydantic Input Validation Schema (FOR BONUS POINTS) ---
# Defines the expected structure and types of incoming prediction requests
class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group", ge=0)
    HouseAge: float = Field(..., description="Median house age in block group", ge=0)
    AveRooms: float = Field(..., description="Average number of rooms per household", ge=0)
    AveBedrms: float = Field(..., description="Average number of bedrooms per household", ge=0)
    Population: float = Field(..., description="Block group population", ge=0)
    AveOccup: float = Field(..., description="Average number of household members", ge=0)
    Latitude: float = Field(..., description="Block group latitude", ge=-90, le=90)
    Longitude: float = Field(..., description="Block group longitude", ge=-180, le=180)

class PredictionInput(BaseModel):
    instances: List[HousingFeatures] # Expects a list of feature sets

# --- API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    PREDICTION_REQUESTS.inc() # Increment counter for every prediction request
    with PREDICTION_LATENCY.time(): # Measure time taken for prediction
        # Log the incoming request
        logger.info(f"Received prediction request.")

        # Check if model and scaler are loaded (service readiness check)
        if model is None or scaler is None:
            logger.error("Model or Scaler not loaded. Service is unhealthy for prediction.")
            return jsonify({"error": "Service is not ready. Model or scaler unavailable."}), 503

        try:
            # Parse incoming JSON data
            raw_json_data = request.get_json(force=True)
            logger.info(f"Raw JSON data received: {raw_json_data}")

            # --- Input Validation (using Pydantic) ---
            try:
                validated_input = PredictionInput(instances=raw_json_data)
                # Convert Pydantic models back to list of dicts for pandas DataFrame
                data_for_df = [instance.dict() for instance in validated_input.instances]
                logger.info("Input data validated successfully with Pydantic.")
            except ValidationError as e:
                logger.error(f"Input validation failed: {e.errors()}")
                # Return 422 Unprocessable Entity for validation errors
                return jsonify({'error': 'Input validation failed', 'details': e.errors()}), 422

            # Convert validated input to pandas DataFrame, ensuring correct feature order
            input_df = pd.DataFrame(data_for_df)
            input_df = input_df[EXPECTED_FEATURES] # Reorder and select features
            logger.info(f"Input DataFrame for prediction:\n{input_df}")

            # Preprocess the input data using the loaded scaler
            input_scaled_array = scaler.transform(input_df)
            input_scaled_df = pd.DataFrame(input_scaled_array, columns=EXPECTED_FEATURES, index=input_df.index)
            logger.info(f"Scaled input DataFrame:\n{input_scaled_df}")

            # Make predictions
            predictions = model.predict(input_scaled_df)
            prediction_output = predictions.tolist() # Convert numpy array to list for JSON response

            # Log prediction output
            logger.info(f"Prediction output: {prediction_output}")
            return jsonify({'predictions': prediction_output})

        except Exception as e:
            logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
            return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify if the service is running and
    if the model and scaler are loaded.
    """
    if model is not None and scaler is not None:
        return jsonify({'status': 'healthy', 'model_loaded': True, 'scaler_loaded': True}), 200
    else:
        logger.warning(f"Health check: Model loaded: {model is not None}, Scaler loaded: {scaler is not None}")
        return jsonify({'status': 'unhealthy', 'model_loaded': (model is not None), 'scaler_loaded': (scaler is not None)}), 503

@app.route('/metrics') # Endpoint for Prometheus to scrape metrics
def metrics():
    """
    Exposes Prometheus-formatted metrics.
    """
    return Response(generate_latest(), mimetype='text/plain')

if __name__ == '__main__':
    logger.info("Starting Flask app in debug mode for local testing (NOT Gunicorn)...")
    app.run(host='0.0.0.0', port=5000, debug=True)