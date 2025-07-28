FROM python:3.10-slim-bookworm

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    openssh-client \
    libgit2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "dvc"

COPY . /app


RUN dvc config cache.dir /dvc_cache
RUN dvc config remote.d_drive_remote.url /dvc_remote
RUN dvc config core.remote d_drive_remote


# Add some debugging output to see if files are present after dvc pull
RUN echo "--- Before DVC Pull (inside container) ---"
RUN ls -l data/raw/ || true # Use || true to prevent build failure if dir is empty/not exists yet
RUN ls -l data/processed/ || true

# Pull DVC-tracked data into the container's workspace (/app/data)
# This uses the DVC cache/remote mounted from your D: drive.
# RUN dvc pull --force

RUN echo "--- After DVC Pull (inside container) ---"
RUN ls -l data/raw/
RUN ls -l data/processed/
RUN ls -l data/processed/scaler.pkl # Explicitly check scaler.pkl

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.main:app"]