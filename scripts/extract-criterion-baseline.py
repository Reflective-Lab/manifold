#!/usr/bin/env python3
"""Extract a compact Criterion baseline into kb/Baselines."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CRITERION = ROOT / "target" / "criterion"
BASELINES = ROOT / "kb" / "Baselines"


def micros(value_ns: float | int | None) -> float:
    if value_ns is None:
        return 0.0
    return round(float(value_ns) / 1000.0, 2)


def point(estimates: dict, key: str) -> float:
    return estimates.get(key, {}).get("point_estimate", 0.0)


def upper(estimates: dict, key: str) -> float:
    return estimates.get(key, {}).get("confidence_interval", {}).get("upper_bound", 0.0)


def benchmark_name(estimates_path: Path) -> str:
    relative = estimates_path.relative_to(CRITERION)
    return "/".join(relative.parts[:-2])


def collect() -> dict[str, dict[str, float]]:
    benchmarks: dict[str, dict[str, float]] = {}
    for estimates_path in sorted(CRITERION.glob("**/new/estimates.json")):
        name = benchmark_name(estimates_path)
        if not name or name.startswith("report/"):
            continue
        estimates = json.loads(estimates_path.read_text())
        benchmarks[name] = {
            "p50_us": micros(point(estimates, "median")),
            "p95_us": micros(upper(estimates, "median")),
            "p99_us": micros(upper(estimates, "mean")),
            "mean_us": micros(point(estimates, "mean")),
            "std_dev_us": micros(point(estimates, "std_dev")),
        }
    return benchmarks


def write_summary(run_id: str, timestamp: str, benchmarks: dict[str, dict[str, float]]) -> None:
    lines = [
        "# Benchmark Baseline Summary",
        "",
        f"**Run:** {run_id} ({timestamp})",
        "",
        "| Benchmark | p50 (us) | p95 (us) | p99 (us) | Mean (us) | StdDev (us) |",
        "|-----------|----------|----------|----------|-----------|-------------|",
    ]
    for name, stats in sorted(benchmarks.items()):
        lines.append(
            "| {name} | {p50_us:.2f} | {p95_us:.2f} | {p99_us:.2f} | {mean_us:.2f} | {std_dev_us:.2f} |".format(
                name=name,
                **stats,
            )
        )
    (BASELINES / "latest-summary.md").write_text("\n".join(lines) + "\n")


def append_trends(run_id: str, timestamp: str, benchmarks: dict[str, dict[str, float]]) -> None:
    trends_path = BASELINES / "trends.csv"
    exists = trends_path.exists()
    with trends_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                ["date", "run_id", "benchmark", "p50_us", "p95_us", "p99_us", "mean_us", "std_dev_us"]
            )
        date = timestamp.split("T", 1)[0]
        for name, stats in sorted(benchmarks.items()):
            writer.writerow(
                [
                    date,
                    run_id,
                    name,
                    stats["p50_us"],
                    stats["p95_us"],
                    stats["p99_us"],
                    stats["mean_us"],
                    stats["std_dev_us"],
                ]
            )


def main() -> int:
    if not CRITERION.exists():
        print(f"No Criterion output found at {CRITERION}")
        return 1
    benchmarks = collect()
    if not benchmarks:
        print(f"No Criterion estimates found under {CRITERION}")
        return 1

    now = datetime.now(timezone.utc)
    timestamp = now.isoformat().replace("+00:00", "Z")
    run_id = now.strftime("%Y%m%d-%H%M%S")
    BASELINES.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": timestamp,
        "run_id": run_id,
        "benchmarks": benchmarks,
    }
    (BASELINES / "latest-baseline.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_summary(run_id, timestamp, benchmarks)
    append_trends(run_id, timestamp, benchmarks)
    print(f"Wrote {len(benchmarks)} benchmark baselines to {BASELINES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
