import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
from sklearn.datasets import fetch_california_housing

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://localhost:5500")
mlflow.set_experiment("California_Housing_Regression")

# Load dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "LinearRegression": LinearRegression()
}

# Train, log, register
client = MlflowClient()
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"{model_name} MSE: {mse}")
        
        # Register
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        registered = mlflow.register_model(model_uri, model_name)
        
        # Promote to production
        client.transition_model_version_stage(
            name=model_name,
            version=1,  # or fetch dynamically if needed
            stage="Production",
            archive_existing_versions=True
        )

print("All models trained, logged, registered, and promoted to Production.")