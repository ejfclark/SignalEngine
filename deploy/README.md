# Deploying SignalEngine on the IONOS VPS

The whole system is: a directory of Parquet files, a Python venv, and two
systemd timers. No database required on the box (the SQL bridge job talks to
Azure only for the market-P/E / bond-yield context while StockIngest still
produces it).

## One-time setup

```bash
sudo useradd -r -m -d /opt/signalengine signalengine
sudo -u signalengine git clone <repo> /opt/signalengine
cd /opt/signalengine
sudo -u signalengine python3 -m venv .venv
sudo -u signalengine .venv/bin/pip install -e .

# secrets
sudo mkdir -p /etc/signalengine
sudo cp deploy/signalengine.env.example /etc/signalengine/signalengine.env
sudo chmod 600 /etc/signalengine/signalengine.env   # then edit with real values

# unix ODBC driver, only if you keep the SQL context bridge:
# curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add - ... (msodbcsql18)

# initial data: either copy the lake from your dev machine (fastest)
rsync -av data/ vps:/opt/signalengine/data/
# ...or rebuild it on the box:
#   signalengine ingest legacy-snapshot && signalengine ingest stocks --backfill && ...

# timers
sudo cp deploy/systemd/*.service deploy/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now signalengine-ingest.timer signalengine-signals.timer
```

## Observe

```bash
systemctl list-timers 'signalengine-*'
journalctl -u signalengine-ingest.service -n 50
cat artifacts/stock_signals.csv
```

## Backup

The lake IS the asset. One line in cron covers it:

```bash
rsync -a --delete /opt/signalengine/data/ /backup/signalengine-data/
# or rclone to any object storage (IONOS S3, Backblaze B2, ...)
```

## Portal (signals web UI)

```bash
.venv/bin/pip install -e '.[portal]'
.venv/bin/python -m signalengine.portal --make-hash   # paste output into the env file
sudo cp deploy/systemd/signalengine-portal.service /etc/systemd/system/
sudo systemctl enable --now signalengine-portal
```

Gunicorn listens on 127.0.0.1:8050 — put nginx or caddy in front for TLS, e.g. caddy:

```
signals.yourdomain.com {
    reverse_proxy 127.0.0.1:8050
}
```

(Don't expose 8050 directly: the login is a single shared password over whatever
transport you give it — TLS is the part that makes that acceptable.)

## Schedule (UTC)

| When | What |
|---|---|
| 21:30/21:40 | StockIngest (C#) market job — still feeds MarketPE/SectorPE/BondYield to SQL |
| 22:15 | `ingest daily` — prices, ETFs, macro, crypto, funding, fundamentals, SQL context |
| 23:00 | `train --rebuild` + `signals` for both assets → `artifacts/*_signals.csv` |

StockIngest's **prices** timers should be disabled once this is live — the lake
supersedes them (`systemctl disable stock-ingest-prices.timer`).
