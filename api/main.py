import mlflow
import mlflow.pyfunc
from flask import Flask, request, jsonify, Response
import os
import logging
import pandas as pd
import joblib
from pydantic import BaseModel, Field, ValidationError
from typing import List

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow Model Configuration
MODEL_NAME = "CaliforniaHousingRegressor"
MODEL_ALIAS = "production"
MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# --- IMPORTANT: Configure MLflow Tracking URI for Docker Container ---
# When MLflow UI is running on your host machine's localhost,
# Docker containers need to use 'host.docker.internal' to reach it.
# Ensure your MLflow UI is running (mlflow ui in host terminal).
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow-tracking-server:5000"
logger.info(f"MLFLOW_TRACKING_URI set to: {os.environ['MLFLOW_TRACKING_URI']}")

# Scaler Path (relative to the container's working directory /app)
SCALER_CONTAINER_PATH = "data/processed/scaler.pkl"

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
        model = mlflow.pyfunc.load_model(model_uri=MLFLOW_MODEL_URI)
        logger.info(f"MLflow model '{MODEL_NAME}' with alias '{MODEL_ALIAS}' loaded successfully.")

        logger.info(f"Loading scaler from path: {SCALER_CONTAINER_PATH}")
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

# Load artifacts when the Flask app starts
# Call this after `app = Flask(__name__)`
with app.app_context():
    load_artifacts()


# --- Pydantic Input Validation Schema (FOR BONUS POINTS) ---
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
    instances: List[HousingFeatures]

# --- API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    logger.info(f"Received prediction request.")

    if model is None or scaler is None:
        logger.error("Model or Scaler not loaded. Service is unhealthy for prediction.")
        return jsonify({"error": "Service is not ready. Model or scaler unavailable."}), 503

    try:
        raw_json_data = request.get_json(force=True)
        logger.info(f"Raw JSON data received: {raw_json_data}")

        try:
            validated_input = PredictionInput(instances=raw_json_data)
            data_for_df = [instance.dict() for instance in validated_input.instances]
            logger.info("Input data validated successfully with Pydantic.")
        except ValidationError as e:
            logger.error(f"Input validation failed: {e.errors()}")
            return jsonify({'error': 'Input validation failed', 'details': e.errors()}), 422

        input_df = pd.DataFrame(data_for_df)
        input_df = input_df[EXPECTED_FEATURES]
        logger.info(f"Input DataFrame for prediction:\n{input_df}")

        input_scaled_array = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled_array, columns=EXPECTED_FEATURES, index=input_df.index)
        logger.info(f"Scaled input DataFrame:\n{input_scaled_df}")

        predictions = model.predict(input_scaled_df)
        prediction_output = predictions.tolist()

        logger.info(f"Prediction output: {prediction_output}")
        return jsonify({'predictions': prediction_output})

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    if model is not None and scaler is not None:
        return jsonify({'status': 'healthy', 'model_loaded': True, 'scaler_loaded': True}), 200
    else:
        logger.warning(f"Health check: Model loaded: {model is not None}, Scaler loaded: {scaler is not None}")
        return jsonify({'status': 'unhealthy', 'model_loaded': (model is not None), 'scaler_loaded': (scaler is not None)}), 503

if __name__ == '__main__':
    logger.info("Starting Flask app in debug mode for local testing (NOT Gunicorn)...")
    app.run(host='0.0.0.0', port=5000, debug=True)