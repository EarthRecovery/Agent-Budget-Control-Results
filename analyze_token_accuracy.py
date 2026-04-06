#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional


def coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_total_tokens(record: Dict[str, Any]) -> Optional[int]:
    total_tokens = coerce_optional_int(record.get("api_total_tokens"))
    if total_tokens is not None:
        return total_tokens

    input_tokens = coerce_optional_int(record.get("api_input_tokens"))
    output_tokens = coerce_optional_int(record.get("api_output_tokens"))
    if input_tokens is None or output_tokens is None:
        return None
    return input_tokens + output_tokens


def compute_total_used_tokens(payload: Any) -> float:
    if not isinstance(payload, list):
        return math.nan

    env_totals: List[int] = []
    for env_record in payload:
        if not isinstance(env_record, dict):
            continue

        env_total_tokens = extract_total_tokens(env_record)
        if env_total_tokens is not None:
            env_totals.append(env_total_tokens)
            continue

        turns = env_record.get("turns", [])
        if not isinstance(turns, list):
            continue

        turn_totals = [
            total_tokens
            for turn in turns
            if isinstance(turn, dict)
            for total_tokens in [extract_total_tokens(turn)]
            if total_tokens is not None
        ]
        if turn_totals:
            env_totals.append(sum(turn_totals))

    return float(sum(env_totals)) if env_totals else math.nan


def compute_relative_error_metrics(
    estimate_tokens: Optional[int],
    actual_tokens: Optional[int],
) -> Dict[str, Optional[float]]:
    if estimate_tokens is None or actual_tokens is None:
        return {"relative_error": None, "diff": None, "error_ratio": None}

    actual = max(1, int(actual_tokens))
    estimate = max(0, int(estimate_tokens))
    diff = int(estimate - actual)
    error_ratio = float(abs(diff) / float(actual))
    return {"relative_error": error_ratio, "diff": diff, "error_ratio": error_ratio}


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


def resolve_default_paths() -> List[Path]:
    candidates = [Path.cwd(), Path.cwd() / "estimation-datasets"]
    paths: List[Path] = []
    seen = set()
    for directory in candidates:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)
    return paths


def summarize_file(path: Path) -> Dict[str, Any]:
    payload = load_payload(path)
    turns = list(iter_turns(payload))
    total_used_tokens = compute_total_used_tokens(payload)

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

        metrics = compute_relative_error_metrics(int(estimate_token), int(actual_token))
        valid_rows.append(
            {
                "estimate_token": int(estimate_token),
                "actual_token": int(actual_token),
                "diff": int(metrics["diff"]),
                "abs_diff": abs(int(metrics["diff"])),
                "error_ratio": float(metrics["error_ratio"]),
                "relative_error": float(metrics["relative_error"]),
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
        "total_used_tokens": total_used_tokens,
        "coverage_rate": (len(valid_rows) / total_turns) if total_turns else 0.0,
        "exact_match_rate": exact_match_rate,
        "overestimate_rate": overestimate_rate,
        "underestimate_rate": underestimate_rate,
        "mean_relative_error": (
            mean(row["relative_error"] for row in valid_rows) if valid_rows else 0.0
        ),
        "median_relative_error": (
            median(row["relative_error"] for row in valid_rows) if valid_rows else 0.0
        ),
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


def format_int_or_nan(value: Any) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return str(int(value))


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"\n== {Path(summary['file']).name} ==")
    print(f"total_turns           : {summary['total_turns']}")
    print(f"valid_turns           : {summary['valid_turns']}")
    print(f"generation_errors     : {summary['generation_error_turns']}")
    print(f"missing_estimates     : {summary['missing_estimate_turns']}")
    print(f"total_used_tokens     : {format_int_or_nan(summary['total_used_tokens'])}")
    print(f"coverage_rate         : {format_percent(summary['coverage_rate'])}")
    print(f"exact_match_rate      : {format_percent(summary['exact_match_rate'])}")
    print(f"overestimate_rate     : {format_percent(summary['overestimate_rate'])}")
    print(f"underestimate_rate    : {format_percent(summary['underestimate_rate'])}")
    print(f"mean_relative_error   : {summary['mean_relative_error']:.4f}")
    print(f"median_relative_error : {summary['median_relative_error']:.4f}")
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
        description="Analyze token estimation relative error from benchmark dialogue JSON files."
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
    return resolve_default_paths()


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
