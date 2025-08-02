#!/bin/bash

echo "Starting deployment..."

# Pull latest image from Docker Hub
docker pull muralikrishnaraparthi/mlops-housing-pipeline-mlops-api:latest

# Stop and remove old container if it exists
docker rm -f mlops-api || true

# Run new container
docker run -d \
  --name mlops-api \
  -p 5001:5000 \
  -v "${PWD}\logs:/app/logs" \
  muralikrishnaraparthi/mlops-housing-pipeline-mlops-api:latest


echo "Deployment complete."
