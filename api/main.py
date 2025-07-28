import os
import logging
import sqlite3
from datetime import datetime
from typing import List

import mlflow
import mlflow.pyfunc
from flask import Flask, request, jsonify, Response
import pandas as pd
import joblib
from pydantic import BaseModel, Field, ValidationError
from prometheus_client import Counter, Summary, generate_latest

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/predictions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- SQLite Logging ---
def log_to_sqlite(input_data: dict, output_data: list):
    conn = sqlite3.connect("logs/predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input TEXT,
            output TEXT
        )
    """)
    timestamp = datetime.utcnow().isoformat()
    cursor.execute("INSERT INTO predictions (timestamp, input, output) VALUES (?, ?, ?)", (
        timestamp, str(input_data), str(output_data)
    ))
    conn.commit()
    conn.close()

# --- MLflow Configuration ---
MODEL_NAME = "CaliforniaHousingRegressor"
MODEL_STAGE = "Production"
MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

os.environ["MLFLOW_TRACKING_URI"] = "http://host.docker.internal:5000"
logger.info(f"MLFLOW_TRACKING_URI set to: {os.environ['MLFLOW_TRACKING_URI']}")

SCALER_CONTAINER_PATH = "data/processed/scaler.pkl"
EXPECTED_FEATURES = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

app = Flask(__name__)
model = None
scaler = None

# --- Prometheus Metrics ---
prediction_requests_total = Counter('prediction_requests_total', 'Total prediction requests')
prediction_success_total = Counter('prediction_success_total', 'Successful predictions')
prediction_error_total = Counter('prediction_error_total', 'Failed predictions')
prediction_latency_seconds = Summary('prediction_latency_seconds', 'Prediction latency in seconds')

# --- Load Artifacts ---
def load_artifacts():
    global model, scaler
    try:
        logger.info("Attempting to load artifacts...")
        logger.info(f"Loading MLflow model from URI: {MLFLOW_MODEL_URI}")
        model = mlflow.pyfunc.load_model(model_uri=MLFLOW_MODEL_URI)
        logger.info("MLflow model loaded successfully.")

        if os.path.exists(SCALER_CONTAINER_PATH):
            scaler = joblib.load(SCALER_CONTAINER_PATH)
            logger.info("Scaler loaded successfully.")
        else:
            logger.error(f"Scaler not found at: {SCALER_CONTAINER_PATH}")
            scaler = None
    except Exception as e:
        logger.critical(f"Failed to load artifacts: {e}", exc_info=True)
        model = None
        scaler = None

with app.app_context():
    load_artifacts()

# --- Input Schemas ---
class HousingFeatures(BaseModel):
    MedInc: float = Field(..., ge=0)
    HouseAge: float = Field(..., ge=0)
    AveRooms: float = Field(..., ge=0)
    AveBedrms: float = Field(..., ge=0)
    Population: float = Field(..., ge=0)
    AveOccup: float = Field(..., ge=0)
    Latitude: float = Field(..., ge=-90, le=90)
    Longitude: float = Field(..., ge=-180, le=180)

class PredictionInput(BaseModel):
    instances: List[HousingFeatures]

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
@prediction_latency_seconds.time()  # ⏱️ latency tracking
def predict():
    logger.info("Received prediction request.")
    prediction_requests_total.inc()

    if model is None or scaler is None:
        logger.error("Model or Scaler not loaded.")
        prediction_error_total.inc()
        return jsonify({"error": "Model or scaler unavailable."}), 503

    try:
        raw_json_data = request.get_json(force=True)
        logger.info(f"Raw JSON: {raw_json_data}")

        try:
            validated_input = PredictionInput(instances=raw_json_data)
            data_for_df = [item.dict() for item in validated_input.instances]
            logger.info("Validated input successfully.")
        except ValidationError as e:
            logger.error(f"Validation failed: {e.errors()}")
            prediction_error_total.inc()
            return jsonify({'error': 'Validation failed', 'details': e.errors()}), 422

        input_df = pd.DataFrame(data_for_df)
        input_df = input_df[EXPECTED_FEATURES]
        logger.info(f"[PREDICT-INPUT] {input_df.to_dict(orient='records')}")

        input_scaled_array = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled_array, columns=EXPECTED_FEATURES, index=input_df.index)
        logger.info(f"[PREDICT-SCALED] {input_scaled_df.to_dict(orient='records')}")

        predictions = model.predict(input_scaled_df)
        prediction_output = predictions.tolist()

        logger.info(f"[PREDICT-OUTPUT] {prediction_output}")
        log_to_sqlite(input_df.to_dict(orient='records'), prediction_output)

        prediction_success_total.inc()
        return jsonify({'predictions': prediction_output})

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        prediction_error_total.inc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500

# --- Health Check ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if model and scaler else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }), 200 if model and scaler else 503

# --- Prometheus Metrics Endpoint ---
@app.route('/metrics', methods=['GET'])
def metrics():
    return Response(generate_latest(), mimetype='text/plain')


# --- Run App ---
if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True)
