"""Ingest jobs: collect data from free/cheap sources into the Parquet lake.

The lake (config [data].parquet_dir) is the source of truth for the engine.
Each job is idempotent — it upserts on natural keys, so re-running after a
failure or overlap is always safe. `signalengine ingest daily` runs the
incremental set; individual jobs support --backfill for full history.
"""
