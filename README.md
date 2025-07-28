# mlops-housing-pipeline
To run, in the root directory, run the command "docker-compose up --build" 

1. FastAPI — The ML Model as a Web Service
	•	What it is: A Python web framework used to expose your ML model as a REST API.
	•	What it does in your setup:
	•	Loads your trained ML model.
	•	Provides an endpoint like /predict that accepts input (via JSON) and returns predictions.
	•	Exposes a /metrics endpoint (via prometheus_client) to show internal metrics like:
	•	Request count
	•	Latency
	•	Error count

✅ Think of FastAPI as the face of your ML model, serving predictions to users or other systems.

⸻

🔷 2. Prometheus — The Metrics Collector
	•	What it is: An open-source system for monitoring and alerting, built for time-series data.
	•	What it does in your setup:
	•	Periodically scrapes your FastAPI’s /metrics endpoint.
	•	Collects data like:
	•	How many times the model is being called
	•	How long predictions take
	•	If any errors are happening

✅ Think of Prometheus as the data-gatherer that checks the health and usage of your ML API.

⸻

🔷 3. Grafana — The Visualization Layer
	•	What it is: An open-source dashboarding tool.
	•	What it does in your setup:
	•	Connects to Prometheus as a data source.
	•	Displays live dashboards for metrics like:
	•	Number of API hits over time
	•	Average response time
	•	Status codes
	•	Any model-related metrics you track
	•	Helps you visualize system performance and detect anomalies or drift.
	http://localhost:3000
	•	Username: admin
	•	Password: admin

	After login, you can:
	•	Add Prometheus as a Data Source:
	•	Go to ⚙️ “Settings” > “Data Sources”
	•	Click “Add data source”
	•	Select Prometheus
	•	Set URL as http://prometheus:9090
	•	Click “Save & Test”
	•	Import Dashboard:
	•	Use predefined JSON dashboard templates
	•	Or create panels showing custom metrics (e.g., from your FastAPI app)

First login will prompt you to change the password — you can set a new one or keep it as is for testing.

✅ Think of Grafana as the monitoring dashboard — it shows you everything happening under the hood of your model.

⸻

🔁 Putting It All Together

Here’s a step-by-step flow:
	1.	🔄 User/API calls FastAPI endpoint → /predict
	2.	⚙️ FastAPI handles the request and returns a prediction.
	3.	📊 FastAPI updates internal metrics (via prometheus_client).
	4.	⏱️ Prometheus scrapes /metrics every 15s (or configured interval).
	5.	📈 Grafana pulls metrics from Prometheus and displays them in real-time dashboards.



🐳 What is Docker?

Docker is a tool that lets you package your code + all dependencies + environment into a single portable unit called a container.

Think of it like creating a mini, isolated Linux machine that runs only your ML app — no matter which system you’re on.

⸻

💡 In Your MLOps Project, Docker Is Doing the Following:

🔹 1. Packaging Your FastAPI App
	•	It takes your FastAPI-based ML service (main.py + model + requirements.txt)
	•	Bundles it into a container using a Dockerfile
	•	This container can now run on any system — Linux, Mac, Windows, local machine, or cloud — without “it works on my machine” issues

⸻

🔹 2. Running Prometheus and Grafana
	•	In your docker-compose.yml, you’re also pulling Docker images for:
	•	Prometheus: from prom/prometheus
	•	Grafana: from grafana/grafana
	•	These images are pre-configured services that spin up immediately — no need to install them manually

⸻

🔹 3. Connecting All Services Together
	•	Docker Compose sets up a mini local cluster:
	•	FastAPI (serving your model)
	•	Prometheus (scraping /metrics)
	•	Grafana (showing dashboards)
	•	They all run in separate containers but communicate with each other on the same Docker network

