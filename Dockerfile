# SignalEngine: one image, two roles.
#   portal (default CMD)  — gunicorn serving the signals dashboard
#   batch                 — `docker compose run --rm engine python -m signalengine.cli ...`
FROM python:3.12-slim

# libgomp1: LightGBM runtime. curl: FRED fetch (their TLS stack tarpits python-requests).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY signalengine ./signalengine
RUN pip install --no-cache-dir ".[portal]"
COPY config.toml ./
COPY universe ./universe

# data lake and artifacts are bind-mounted volumes:
#   /app/data/parquet  /app/artifacts
EXPOSE 8050
CMD ["gunicorn", "-b", "0.0.0.0:8050", "-w", "2", "--access-logfile", "-", "signalengine.portal:app"]
