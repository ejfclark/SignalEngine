"""Signals portal: a small Flask app that reads artifacts/*.csv and shows
today's signals with action badges behind a login.

Auth is deliberately simple for now: one user, password checked against a
salted hash — both from environment (.env):

    PORTAL_USER=ed
    PORTAL_PASSWORD_HASH=<output of: python -m signalengine.portal --make-hash>
    PORTAL_SECRET_KEY=<any long random string; signs the session cookie>

Run locally:    python -m signalengine.portal          (http://127.0.0.1:8050)
Run on VPS:     gunicorn -b 127.0.0.1:8050 signalengine.portal:app
                (behind nginx/caddy for TLS; see deploy/README.md)
"""

from __future__ import annotations

import hmac
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, redirect, render_template_string, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from .config import load_config

cfg = load_config()
app = Flask(__name__)
app.secret_key = os.environ.get("PORTAL_SECRET_KEY", "dev-only-change-me")

STRONG_THRESHOLD = 0.70
WATCH_THRESHOLD = 0.55
STALE_DAYS = 4  # data older than this gets a loud warning tile


def _login_ok(username: str, password: str) -> bool:
    expected_user = os.environ.get("PORTAL_USER", "")
    password_hash = os.environ.get("PORTAL_PASSWORD_HASH", "")
    if not expected_user or not password_hash:
        return False
    return hmac.compare_digest(username, expected_user) and check_password_hash(
        password_hash, password
    )


def _load_signals(tag: str, short: bool = False) -> pd.DataFrame | None:
    path = cfg.artifacts_dir / f"{tag}_signals.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df["action"] = "NONE"
    if short:
        # Shorts trade at the strict threshold only — per the combined-book bench.
        df.loc[df["probability"] >= STRONG_THRESHOLD, "action"] = "SHORT"
    else:
        df.loc[df["probability"] >= WATCH_THRESHOLD, "action"] = "WATCH"
        df.loc[df["probability"] >= STRONG_THRESHOLD, "action"] = "BUY"
    return df.sort_values("probability", ascending=False)


def _asset_view(tag: str, title: str | None = None, short: bool = False) -> dict | None:
    df = _load_signals(tag, short)
    if df is None or df.empty:
        return None
    asof = df["date"].max()
    age_days = (pd.Timestamp.now().normalize() - asof.normalize()).days
    return {
        "name": title or tag,
        "asof": asof.date().isoformat(),
        "stale": age_days > STALE_DAYS,
        "age_days": age_days,
        "n_buy": int(df["action"].isin(("BUY", "SHORT")).sum()),
        "n_watch": int((df["action"] == "WATCH").sum()),
        "top_prob": float(df["probability"].max()),
        "rows": df[df["action"] != "NONE"].head(25).to_dict("records"),
        "mtime": datetime.fromtimestamp(
            (cfg.artifacts_dir / f"{tag}_signals.csv").stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M"),
    }


BASE_CSS = """
:root {
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --good: #006300; --good-badge: #0ca30c; --warn: #fab219; --crit: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --good: #0ca30c;
  }
}
* { box-sizing: border-box; margin: 0; }
body { background: var(--page); color: var(--ink);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
       padding: 24px; max-width: 1080px; margin: 0 auto; }
h1 { font-size: 20px; margin-bottom: 4px; }
h2 { font-size: 16px; margin: 28px 0 10px; text-transform: capitalize; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
.tiles { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0 4px; }
.tile { background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 12px 16px; min-width: 130px; }
.tile .v { font-size: 24px; font-weight: 650; }
.tile .l { font-size: 12px; color: var(--ink-2); }
.tile.warn { border-color: var(--crit); }
.tile.warn .v { color: var(--crit); }
table { width: 100%; border-collapse: collapse; background: var(--surface);
        border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
th { text-align: left; font-size: 12px; color: var(--ink-2); font-weight: 600;
     padding: 8px 12px; border-bottom: 1px solid var(--grid); }
td { padding: 7px 12px; border-bottom: 1px solid var(--grid);
     font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; }
.badge { display: inline-block; font-size: 12px; font-weight: 650;
         padding: 1px 8px; border-radius: 10px; color: #fff; }
.badge.buy { background: var(--good-badge); }
.badge.watch { background: #8a6407; }
.badge.short { background: var(--crit); }
.stop { color: var(--crit); } .target { color: var(--good); }
.empty { color: var(--muted); padding: 16px; background: var(--surface);
         border: 1px solid var(--border); border-radius: 8px; }
form.login { max-width: 320px; margin: 12vh auto; background: var(--surface);
             border: 1px solid var(--border); border-radius: 10px; padding: 24px; }
form.login label { display: block; font-size: 13px; color: var(--ink-2); margin: 10px 0 4px; }
form.login input { width: 100%; padding: 8px 10px; border: 1px solid var(--grid);
                   border-radius: 6px; background: var(--page); color: var(--ink); }
form.login button { margin-top: 16px; width: 100%; padding: 9px; border: 0;
                    border-radius: 6px; background: #2a78d6; color: #fff;
                    font-weight: 650; cursor: pointer; }
.err { color: var(--crit); font-size: 13px; margin-top: 10px; }
a.out { float: right; font-size: 13px; color: var(--muted); }
"""

LOGIN_HTML = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SignalEngine</title><style>{{ css }}</style>
<form class="login" method="post">
  <h1>SignalEngine</h1>
  <label>Username</label><input name="username" autofocus autocomplete="username">
  <label>Password</label><input name="password" type="password" autocomplete="current-password">
  <button>Sign in</button>
  {% if error %}<div class="err">{{ error }}</div>{% endif %}
</form>"""

DASH_HTML = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SignalEngine — signals</title><style>{{ css }}</style>
<a class="out" href="{{ url_for('logout') }}">sign out</a>
<h1>SignalEngine</h1>
<div class="sub">Swing signals — triple-barrier model &middot; BUY &ge; {{ strong }} probability,
WATCH &ge; {{ watch }} &middot; act at next open, stop/target as shown</div>

{% for a in assets %}
  <h2>{{ a.name }}</h2>
  <div class="tiles">
    <div class="tile {{ 'warn' if a.stale }}">
      <div class="v">{{ a.asof }}</div>
      <div class="l">{{ '⚠ data %d days old' % a.age_days if a.stale else 'data as of' }}</div>
    </div>
    <div class="tile"><div class="v">{{ a.n_buy }}</div><div class="l">BUY signals</div></div>
    <div class="tile"><div class="v">{{ a.n_watch }}</div><div class="l">watch list</div></div>
    <div class="tile"><div class="v">{{ '%.0f%%' % (a.top_prob * 100) }}</div><div class="l">top probability</div></div>
  </div>
  {% if a.rows %}
  <table>
    <tr><th>Action</th><th>Ticker</th><th class="num">Prob</th><th class="num">Close</th>
        <th class="num">Stop</th><th class="num">Target</th><th class="num">R:R</th>
        <th class="num">Horizon</th>{% if a.rows[0].sector is defined %}<th>Sector</th>{% endif %}</tr>
    {% for r in a.rows %}
    <tr>
      <td><span class="badge {{ {'BUY': 'buy', 'SHORT': 'short'}.get(r.action, 'watch') }}">
          {{ {'BUY': '▲ BUY', 'SHORT': '▼ SHORT'}.get(r.action, '◔ WATCH') }}</span></td>
      <td><b>{{ r.ticker }}</b></td>
      <td class="num">{{ '%.0f%%' % (r.probability * 100) }}</td>
      <td class="num">{{ '%.2f' % r.close }}</td>
      <td class="num stop">{{ '%.2f' % r.stop }} ({{ '%.1f%%' % (r.stop_pct * 100) }})</td>
      <td class="num target">{{ '%.2f' % r.target }} (+{{ '%.1f%%' % (r.target_pct * 100) }})</td>
      <td class="num">{{ '%.1f' % r.reward_risk }}</td>
      <td class="num">{{ r.horizon_days }}d</td>
      {% if r.sector is defined %}<td>{{ r.sector if r.sector == r.sector else '—' }}</td>{% endif %}
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <div class="empty">No signals above {{ watch }} — the setup isn't there right now. That's the model doing its job.</div>
  {% endif %}
{% else %}
  <div class="empty">No signal files found in artifacts/ yet — run <code>signalengine signals</code>.</div>
{% endfor %}
{% if ledger %}
<h2>paper book — live vs backtest</h2>
<table>
  <tr><th>Book</th><th class="num">Open</th><th class="num">Closed</th>
      <th class="num">Hit rate</th><th class="num">Live expectancy</th><th class="num">Backtest expects</th></tr>
  {% for r in ledger.rows %}
  <tr>
    <td>{{ r.book }}</td>
    <td class="num">{{ r.open }}</td>
    <td class="num">{{ r.closed }}</td>
    <td class="num">{{ '%.0f%%' % (r.hit * 100) if r.hit is not none else '—' }}</td>
    <td class="num">{{ '%+.2f%%' % (r.expectancy * 100) if r.expectancy is not none else '—' }}</td>
    <td class="num">{{ '%+.2f%%' % (r.expected * 100) }}</td>
  </tr>
  {% endfor %}
</table>
<div class="sub" style="margin-top:6px">Virtual positions from nightly signals, filled and closed by the
same rules the model was trained on. This table is the go/no-go evidence for real money.</div>
{% endif %}

<div class="sub" style="margin-top:20px">
  {% for a in assets %}{{ a.name }} file updated {{ a.mtime }}{{ ' · ' if not loop.last }}{% endfor %}
</div>"""


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if _login_ok(request.form.get("username", ""), request.form.get("password", "")):
            session["user"] = request.form["username"]
            return redirect(url_for("dashboard"))
        error = "Wrong username or password."
    return render_template_string(LOGIN_HTML, css=BASE_CSS, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def _ledger_view() -> dict | None:
    """Paper book: live evidence vs backtest expectation, straight off the ledger."""
    path = cfg.root / "data" / "ledger.parquet"
    if not path.is_file():
        return None
    book = pd.read_parquet(path)
    if book.empty:
        return None
    done = book[book["status"].isin(("target", "stop", "timeout"))]
    rows = []
    for tag, expected in (("crypto", 0.0085), ("crypto-short", 0.0096), ("stock", 0.003)):
        fin = done[done["asset"] == tag]
        n_open = int(book[(book["asset"] == tag)
                          & book["status"].isin(("pending", "open"))].shape[0])
        rows.append({
            "book": tag, "open": n_open, "closed": len(fin),
            "hit": (fin["net_return"] > 0).mean() if len(fin) else None,
            "expectancy": fin["net_return"].mean() if len(fin) else None,
            "expected": expected,
        })
    return {"rows": rows, "n_total": len(book)}


@app.route("/")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    assets = [v for v in (
        _asset_view("stock", "stocks — long"),
        _asset_view("crypto", "crypto — long"),
        _asset_view("crypto-short", "crypto — short", short=True),
    ) if v]
    return render_template_string(
        DASH_HTML, css=BASE_CSS, assets=assets, ledger=_ledger_view(),
        strong=STRONG_THRESHOLD, watch=WATCH_THRESHOLD,
    )


def main() -> None:
    if "--make-hash" in sys.argv:
        import getpass

        pw = getpass.getpass("Password to hash: ")
        print("\nAdd these to .env:")
        print(f"PORTAL_PASSWORD_HASH={generate_password_hash(pw)}")
        print("PORTAL_USER=<your username>")
        print(f"PORTAL_SECRET_KEY={os.urandom(24).hex()}")
        return
    app.run(host="127.0.0.1", port=8050, debug=False)


if __name__ == "__main__":
    main()
