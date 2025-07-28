
# 🏠 MLOps Pipeline: California Housing Price Prediction

This project demonstrates a complete **MLOps workflow** for deploying and monitoring a machine learning model that predicts housing prices using the California Housing dataset.

---

## 🚀 Project Structure

```
mlops-housing-pipeline/
├── api/
│   └── main.py               # Flask API serving the model
├── train.py                  # MLflow-based training and model logging
├── retrain_trigger.py        # Auto-retraining on new data detection
├── deploy.sh                 # Deployment script using Docker
├── prometheus.yml            # Prometheus scrape config
├── docker-compose.yml        # Compose stack with API, Prometheus, Grafana
├── requirements.txt
├── data/
│   └── new_data/             # Drop new CSV files here to trigger retrain
├── .github/
│   └── workflows/
│       └── ci-cd.yml         # GitHub Actions pipeline
```

---

## ⚙️ Features

### ✅ Model Serving with Flask
- Loads model from MLflow Registry (Production stage)
- Applies preprocessing with a stored `scaler.pkl`
- Input validation via `pydantic`

### ✅ CI/CD with GitHub Actions
- Linting with `flake8`, testing with `pytest`
- Builds Docker image and pushes to Docker Hub
- Optional deployment via `deploy.sh`

### ✅ Monitoring with Prometheus & Grafana
- `/metrics` endpoint exposes:
  - Total requests
  - Success & error counts
  - Request latency
- Grafana visualizes real-time API performance

### ✅ Logging
- File logs: `logs/predictions.log`
- Structured SQLite logs: `logs/predictions.db`

### ✅ Automatic Retraining Trigger
- `retrain_trigger.py` monitors `data/new_data/`
- New `.csv` triggers model retraining via `train.py`
- Updated model pushed to MLflow Registry

---

## 🔄 Usage Guide

### 🚢 Run with Docker Compose
```bash
docker-compose up --build
```

- API: [http://localhost:5001](http://localhost:5001)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3000](http://localhost:3000)
  - Default login: `admin / admin`

### 🔁 Retraining Trigger
Start the retraining watcher in another terminal:
```bash
python retrain_trigger.py
```
Drop a `.csv` into `data/new_data/` to trigger automatic retraining.

---

## 📦 API Endpoints

| Endpoint        | Method | Description                         |
|-----------------|--------|-------------------------------------|
| `/predict`      | POST   | Predict housing price from features |
| `/health`       | GET    | Check model/scaler health status    |
| `/metrics`      | GET    | Prometheus metrics endpoint         |

---

## 📈 Sample Prometheus Metrics

```
# HELP prediction_requests_total Total prediction requests
# HELP prediction_latency_seconds Prediction latency in seconds
# HELP prediction_success_total Successful predictions
# HELP prediction_error_total Failed predictions
```

---

## ✅ Technologies Used

- **MLflow** – Model tracking & registry
- **Flask** – REST API for model inference
- **Prometheus** – Metrics scraping and monitoring
- **Grafana** – Dashboards for visualization
- **Docker + GitHub Actions** – CI/CD pipeline
- **SQLite + Logging** – Lightweight persistent logging
- **Watchdog** – File-system based retrain trigger

---

## 📌 Requirements

- Python 3.10
- Docker & Docker Compose
- MLflow Tracking Server running at `host.docker.internal:5000`

---

## 🧠 Future Work

- Add authentication for sensitive endpoints
- Support versioned model rollbacks
- Integrate Slack/email alerts for failed retraining

---

## 📜 License

MIT License © 2025 [Your Name or Organization]
