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

COPY . /app
RUN mkdir -p data/raw data/processed
RUN dvc pull --force

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.main:app"]