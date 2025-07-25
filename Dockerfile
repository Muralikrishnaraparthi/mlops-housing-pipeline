# Use an official Python 3.10 runtime based on Debian Bookworm (Debian 12)
FROM python:3.10-slim-bookworm # <--- CHANGE THIS LINE TO PYTHON 3.10

# ... (rest of the Dockerfile remains the same as our last working version)

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker's build cache
COPY requirements.txt .

# --- Install system-level build dependencies, git, openssh-client, AND libgit2-dev ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    openssh-client \
    libgit2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC separately
RUN pip install "dvc[s3]"

# Copy the rest of the application code into the container
COPY . /app

# Ensure data directories exist inside the container for DVC to materialize files
RUN mkdir -p data/raw data/processed

# IMPORTANT: Pull DVC-versioned data/artifacts
RUN dvc pull

# Expose the port the Flask app will run on
EXPOSE 5000

# Command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.main:app"]