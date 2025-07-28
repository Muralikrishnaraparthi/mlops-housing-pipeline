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
RUN pip install "dvc[s3]"

# Set DVC_GLOBAL_CACHE_DIR here for consistency, it will be overridden by docker-compose if set there
ENV DVC_GLOBAL_CACHE_DIR=/dvc_cache

COPY . /app


RUN dvc config cache.dir /dvc_cache
RUN dvc config remote.d_drive_remote.url /dvc_remote
RUN dvc config core.remote d_drive_remote
RUN dvc pull

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.main:app"]