from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import logging
from prometheus_client import start_http_server, Summary, Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI()

# Start Prometheus metrics server
REQUEST_COUNT = Counter('prediction_requests_total', 'Total number of prediction requests')
REQUEST_LATENCY = Summary('request_latency_seconds', 'Time spent handling request')

# Example input schema
class InputData(BaseModel):
    features: list

# Load model
mlflow.set_tracking_uri("file:/app/mlruns")
model = mlflow.sklearn.load_model("/app/models/LinearRegression.pkl")

@app.post("/predict")
async def predict(data: InputData):
    REQUEST_COUNT.inc()
    start_time = time.time()

    arr = np.array(data.features).reshape(1, -1)
    prediction = model.predict(arr).tolist()

    duration = time.time() - start_time
    REQUEST_LATENCY.observe(duration)

    return {"prediction": prediction}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)