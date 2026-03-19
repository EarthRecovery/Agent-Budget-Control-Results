#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional


def compute_estimation_factor(estimate_tokens: Optional[int], actual_tokens: Optional[int]) -> Dict[str, Optional[float]]:
    if estimate_tokens is None or actual_tokens is None:
        return {"factor": 0.0, "diff": None, "error_ratio": None}

    actual = max(1, int(actual_tokens))
    estimate = max(0, int(estimate_tokens))
    diff = int(estimate - actual)
    error_ratio = float(abs(diff) / float(actual))
    factor = float(max(0.0, 1.0 - error_ratio))
    return {"factor": factor, "diff": diff, "error_ratio": error_ratio}


def iter_turns(payload: Any) -> Iterable[Dict[str, Any]]:
    if not isinstance(payload, list):
        return
    for env_record in payload:
        if not isinstance(env_record, dict):
            continue
        turns = env_record.get("turns", [])
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            merged = dict(turn)
            merged.setdefault("env_id", env_record.get("env_id"))
            merged.setdefault("tag", env_record.get("tag"))
            yield merged


def load_payload(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_file(path: Path) -> Dict[str, Any]:
    payload = load_payload(path)
    turns = list(iter_turns(payload))

    total_turns = len(turns)
    generation_error_turns = 0
    missing_estimate_turns = 0
    valid_rows: List[Dict[str, Any]] = []

    for turn in turns:
        generation_error = turn.get("generation_error")
        generation_success = turn.get("generation_success")
        estimate_token = turn.get("estimate_token")
        actual_token = turn.get("actual_token")

        if generation_error is not None or generation_success is False:
            generation_error_turns += 1
            continue

        if estimate_token is None:
            missing_estimate_turns += 1
            continue

        if actual_token is None:
            continue

        metrics = compute_estimation_factor(int(estimate_token), int(actual_token))
        valid_rows.append(
            {
                "estimate_token": int(estimate_token),
                "actual_token": int(actual_token),
                "diff": int(metrics["diff"]),
                "abs_diff": abs(int(metrics["diff"])),
                "error_ratio": float(metrics["error_ratio"]),
                "accuracy": float(metrics["factor"]),
            }
        )

    exact_match_rate = (
        sum(1 for row in valid_rows if row["diff"] == 0) / len(valid_rows)
        if valid_rows
        else 0.0
    )
    overestimate_rate = (
        sum(1 for row in valid_rows if row["diff"] > 0) / len(valid_rows)
        if valid_rows
        else 0.0
    )
    underestimate_rate = (
        sum(1 for row in valid_rows if row["diff"] < 0) / len(valid_rows)
        if valid_rows
        else 0.0
    )

    return {
        "file": str(path),
        "total_turns": total_turns,
        "valid_turns": len(valid_rows),
        "generation_error_turns": generation_error_turns,
        "missing_estimate_turns": missing_estimate_turns,
        "coverage_rate": (len(valid_rows) / total_turns) if total_turns else 0.0,
        "exact_match_rate": exact_match_rate,
        "overestimate_rate": overestimate_rate,
        "underestimate_rate": underestimate_rate,
        "mean_accuracy": mean(row["accuracy"] for row in valid_rows) if valid_rows else 0.0,
        "median_accuracy": median(row["accuracy"] for row in valid_rows) if valid_rows else 0.0,
        "mean_abs_error": mean(row["abs_diff"] for row in valid_rows) if valid_rows else 0.0,
        "median_abs_error": median(row["abs_diff"] for row in valid_rows) if valid_rows else 0.0,
        "mean_signed_error": mean(row["diff"] for row in valid_rows) if valid_rows else 0.0,
        "mean_abs_error_ratio": mean(row["error_ratio"] for row in valid_rows) if valid_rows else 0.0,
        "median_abs_error_ratio": median(row["error_ratio"] for row in valid_rows) if valid_rows else 0.0,
        "within_10pct_rate": (
            sum(1 for row in valid_rows if row["error_ratio"] <= 0.10) / len(valid_rows)
            if valid_rows
            else 0.0
        ),
        "within_20pct_rate": (
            sum(1 for row in valid_rows if row["error_ratio"] <= 0.20) / len(valid_rows)
            if valid_rows
            else 0.0
        ),
        "within_50pct_rate": (
            sum(1 for row in valid_rows if row["error_ratio"] <= 0.50) / len(valid_rows)
            if valid_rows
            else 0.0
        ),
    }


def format_percent(value: float) -> str:
    return f"{value * 100.0:6.2f}%"


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"\n== {Path(summary['file']).name} ==")
    print(f"total_turns           : {summary['total_turns']}")
    print(f"valid_turns           : {summary['valid_turns']}")
    print(f"generation_errors     : {summary['generation_error_turns']}")
    print(f"missing_estimates     : {summary['missing_estimate_turns']}")
    print(f"coverage_rate         : {format_percent(summary['coverage_rate'])}")
    print(f"exact_match_rate      : {format_percent(summary['exact_match_rate'])}")
    print(f"overestimate_rate     : {format_percent(summary['overestimate_rate'])}")
    print(f"underestimate_rate    : {format_percent(summary['underestimate_rate'])}")
    print(f"mean_accuracy         : {summary['mean_accuracy']:.4f}")
    print(f"median_accuracy       : {summary['median_accuracy']:.4f}")
    print(f"mean_abs_error        : {summary['mean_abs_error']:.2f}")
    print(f"median_abs_error      : {summary['median_abs_error']:.2f}")
    print(f"mean_signed_error     : {summary['mean_signed_error']:.2f}")
    print(f"mean_abs_error_ratio  : {summary['mean_abs_error_ratio']:.4f}")
    print(f"median_abs_error_ratio: {summary['median_abs_error_ratio']:.4f}")
    print(f"within_10pct_rate     : {format_percent(summary['within_10pct_rate'])}")
    print(f"within_20pct_rate     : {format_percent(summary['within_20pct_rate'])}")
    print(f"within_50pct_rate     : {format_percent(summary['within_50pct_rate'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token estimation accuracy from benchmark dialogue JSON files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="JSON files to analyze. If omitted, analyze all *.json files in the current directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON instead of a human-readable report.",
    )
    return parser.parse_args()


def resolve_paths(raw_paths: List[str]) -> List[Path]:
    if raw_paths:
        return [Path(path).expanduser().resolve() for path in raw_paths]
    return sorted(Path.cwd().glob("*.json"))


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args.paths)
    if not paths:
        raise SystemExit("No JSON files found to analyze.")

    summaries = [summarize_file(path) for path in paths]
    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return 0

    for summary in summaries:
        print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
