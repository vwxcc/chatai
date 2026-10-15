FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000 \
    DATA_DIR=/app/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY main.py /app/main.py
COPY tests /app/tests
RUN mkdir -p /app/data

# Render forwards external HTTP traffic to the port configured in PORT.
EXPOSE 10000

# exec ensures uvicorn receives shutdown/restart signals directly.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \${PORT:-10000}"]
