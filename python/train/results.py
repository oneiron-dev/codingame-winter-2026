from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CURRENT_ACCEPTANCE_VERSION = 3

WEIGHTS = {
    "heldout_body_diff": 0.45,
    "heldout_win_margin": 0.25,
    "shadow_body_diff": 0.10,
    "validation_correlation": 0.15,
    "validation_score": 0.10,
}

GATES = {
    "later_turn_p99_ms_max": 45.0,
    "validation_mae_max": 0.75,
}


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: list[str]


def compute_composite(metrics: dict[str, Any]) -> float:
    return float(sum(WEIGHTS[key] * float(metrics.get(key, 0.0)) for key in WEIGHTS))


def check_gates(metrics: dict[str, Any]) -> GateResult:
    failures: list[str] = []
    for gate, threshold in GATES.items():
        metric_key = gate.replace("_max", "")
        if metric_key not in metrics:
            continue
        value = float(metrics[metric_key])
        if value > threshold:
            failures.append(f"{metric_key} {value:.4f} > {threshold:.4f}")
    return GateResult(not failures, failures)


def ensure_schema(path: str | Path) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                description TEXT NOT NULL,
                composite_score REAL NOT NULL,
                acceptance_version INTEGER NOT NULL DEFAULT 1,
                metrics_json TEXT NOT NULL,
                failures_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(experiments)").fetchall()
        }
        if "acceptance_version" not in columns:
            conn.execute(
                "ALTER TABLE experiments ADD COLUMN acceptance_version INTEGER NOT NULL DEFAULT 1"
            )
        conn.execute(
            """
            UPDATE experiments
            SET status = 'legacy'
            WHERE status = 'accepted' AND acceptance_version < ?
            """,
            (CURRENT_ACCEPTANCE_VERSION,),
        )


def append_result(
    path: str | Path,
    *,
    name: str,
    status: str,
    description: str,
    metrics: dict[str, Any],
    failures: list[str],
    acceptance_version: int = CURRENT_ACCEPTANCE_VERSION,
) -> None:
    ensure_schema(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO experiments (
                name,
                status,
                description,
                composite_score,
                acceptance_version,
                metrics_json,
                failures_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                status,
                description,
                compute_composite(metrics),
                acceptance_version,
                json.dumps(metrics, sort_keys=True),
                json.dumps(failures, sort_keys=True),
            ),
        )
