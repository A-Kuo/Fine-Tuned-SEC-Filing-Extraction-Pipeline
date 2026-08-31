FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ci.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY serving ./serving
COPY training ./training
COPY evaluation ./evaluation
COPY monitoring ./monitoring
COPY schemas ./schemas
COPY scripts ./scripts
COPY config.yaml ./
COPY Makefile ./
COPY .env.example ./

EXPOSE 8000

CMD ["python", "-m", "serving.api"]