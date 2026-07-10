"""Configuration: config.toml + .env, with environment variables winning."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, no expansion. Existing env wins."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class LabelConfig:
    horizon_days: int = 10
    target_atr_mult: float = 3.0
    stop_atr_mult: float = 1.5
    # Meta-labeling: pandas query defining candidate rows; empty = all rows.
    candidate_query: str = ""


@dataclass
class CvConfig:
    n_folds: int = 5
    purge_days: int = 10
    # Lockbox holdout: bench/experiment runs never see rows whose trade could
    # touch dates >= lockbox_start (ISO date string; "" disables). Evaluated
    # ONCE per frozen system via `signalengine lockbox-eval`, then re-carved.
    lockbox_start: str = ""


@dataclass
class ModelConfig:
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 50
    # Isotonic calibration: fit on the last calibration_frac of each train
    # window (out-of-time for the model), so a 0.65 probability means the same
    # thing in every fold. Off until benched in.
    calibrate: bool = False
    calibration_frac: float = 0.2


@dataclass
class BacktestConfig:
    probability_threshold: float = 0.60
    fee_bps: float = 10.0
    slippage_bps: float = 10.0
    max_positions: int = 10
    sizing: str = "equal"          # "vol" = risk_pct of equity per trade via stop distance
    risk_pct: float = 0.01
    top_n: int = 0                 # 0 = no per-day cap
    gate_column: str = ""          # e.g. "breadth_20d": skip entries when below gate_min
    gate_min: float = 0.0


@dataclass
class Config:
    root: Path
    source: str = "sql"
    parquet_dir: Path = Path("data/parquet")
    artifacts_dir: Path = Path("artifacts")
    # Which universe file the stock ingest reads: "stocks" (full, Tiingo Power)
    # or "stocks-core" (fits the free tier's 500-unique-symbols/month cap).
    stocks_universe: str = "stocks"
    labels: LabelConfig = field(default_factory=LabelConfig)
    labels_overrides: dict = field(default_factory=dict)  # per-book [labels.<tag>]
    cv: CvConfig = field(default_factory=CvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    backtest_overrides: dict = field(default_factory=dict)  # per-asset [backtest.<asset>]
    signal_threshold: float = 0.55

    def backtest_for(self, tag: str) -> BacktestConfig:
        """Base [backtest] with [backtest.<tag>] overrides. Exact tag first
        ('crypto-short'), then the base asset ('crypto') — a short book only
        inherits from its asset when it has no section of its own."""
        override = self.backtest_overrides.get(tag)
        if override is None:
            override = self.backtest_overrides.get(tag.split("-")[0], {})
        merged = {**self.backtest.__dict__, **override}
        return BacktestConfig(**merged)

    def labels_for(self, tag: str) -> LabelConfig:
        """Per-book barriers/horizon/candidate filter, e.g. [labels.crypto],
        [labels.stock]. `tag` is the book tag ('crypto', 'stock-short', ...)."""
        merged = {**self.labels.__dict__, **self.labels_overrides.get(tag, {})}
        return LabelConfig(**merged)

    @property
    def connection_string(self) -> str:
        conn = os.environ.get("EDSTOCK_CONN", "")
        if not conn:
            raise RuntimeError(
                "EDSTOCK_CONN is not set. Copy .env.example to .env and fill in the "
                "EdStock connection string, or export EDSTOCK_CONN."
            )
        return conn


def load_config(config_path: str | Path | None = None) -> Config:
    """Load config.toml relative to the project root (directory containing it)."""
    if config_path:
        path = Path(config_path).resolve()
    else:
        # Walk up from cwd looking for config.toml so the CLI works from anywhere.
        cur = Path.cwd()
        for candidate in (cur, *cur.parents):
            if (candidate / "config.toml").is_file():
                break
        else:
            candidate = Path(__file__).resolve().parent.parent
        path = candidate / "config.toml"

    root = path.parent
    load_dotenv(root / ".env")

    raw: dict = {}
    if path.is_file():
        raw = tomllib.loads(path.read_text(encoding="utf-8"))

    data = raw.get("data", {})
    backtest_raw = dict(raw.get("backtest", {}))
    overrides = {k: backtest_raw.pop(k) for k in list(backtest_raw)
                 if isinstance(backtest_raw[k], dict)}
    labels_raw = dict(raw.get("labels", {}))
    labels_overrides = {k: labels_raw.pop(k) for k in list(labels_raw)
                        if isinstance(labels_raw[k], dict)}
    cfg = Config(
        root=root,
        source=data.get("source", "sql"),
        parquet_dir=root / data.get("parquet_dir", "data/parquet"),
        artifacts_dir=root / data.get("artifacts_dir", "artifacts"),
        stocks_universe=data.get("stocks_universe", "stocks"),
        labels=LabelConfig(**labels_raw),
        labels_overrides=labels_overrides,
        cv=CvConfig(**raw.get("cv", {})),
        model=ModelConfig(**raw.get("model", {})),
        backtest=BacktestConfig(**backtest_raw),
        backtest_overrides=overrides,
        signal_threshold=raw.get("signals", {}).get("probability_threshold", 0.55),
    )
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return cfg
