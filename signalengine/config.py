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


@dataclass
class CvConfig:
    n_folds: int = 5
    purge_days: int = 10


@dataclass
class ModelConfig:
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 50


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
    labels: LabelConfig = field(default_factory=LabelConfig)
    cv: CvConfig = field(default_factory=CvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    backtest_overrides: dict = field(default_factory=dict)  # per-asset [backtest.<asset>]
    signal_threshold: float = 0.55

    def backtest_for(self, asset: str) -> BacktestConfig:
        """Base [backtest] settings with any [backtest.<asset>] overrides applied —
        portfolio rules are evidence-per-asset, so they are configured per asset."""
        merged = {**self.backtest.__dict__, **self.backtest_overrides.get(asset, {})}
        return BacktestConfig(**merged)

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
    cfg = Config(
        root=root,
        source=data.get("source", "sql"),
        parquet_dir=root / data.get("parquet_dir", "data/parquet"),
        artifacts_dir=root / data.get("artifacts_dir", "artifacts"),
        labels=LabelConfig(**raw.get("labels", {})),
        cv=CvConfig(**raw.get("cv", {})),
        model=ModelConfig(**raw.get("model", {})),
        backtest=BacktestConfig(**backtest_raw),
        backtest_overrides=overrides,
        signal_threshold=raw.get("signals", {}).get("probability_threshold", 0.55),
    )
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return cfg
