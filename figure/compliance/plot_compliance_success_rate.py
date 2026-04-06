#!/usr/bin/env python3
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
DATASET_DIR = PROJECT_ROOT / "compliance-datasets"
OUTPUT_CSV = ROOT / "compliance_success_rates.csv"
OUTPUT_PNG = ROOT / "compliance_success_rates.png"
JSON_PATTERN = "*_eval_compliance_dialogues.json"


@dataclass(frozen=True)
class BudgetSpec:
    budget_type: str
    title: str
    x_label: str
    limit_fields: Tuple[str, ...]
    row_within_fields: Tuple[str, ...]
    turn_within_fields: Tuple[str, ...]
    success_within_fields: Tuple[str, ...]


BUDGET_SPECS: Dict[str, BudgetSpec] = {
    "token": BudgetSpec(
        budget_type="token",
        title="Token Budget",
        x_label="Token Limit",
        limit_fields=("budget_token", "compliance_token_limit"),
        row_within_fields=("within_token_limit",),
        turn_within_fields=("within_token_limit",),
        success_within_fields=("success_within_token_limit",),
    ),
    "turn": BudgetSpec(
        budget_type="turn",
        title="Turn Budget",
        x_label="Turn Limit",
        limit_fields=("budget_turn", "compliance_turn_limit"),
        row_within_fields=("within_turn_limit",),
        turn_within_fields=("within_turn_limit",),
        success_within_fields=("success_within_turn_limit",),
    ),
    "toolcall": BudgetSpec(
        budget_type="toolcall",
        title="Tool Call Budget",
        x_label="Tool Call Limit",
        limit_fields=("budget_toolcall", "compliance_toolcall_limit"),
        row_within_fields=("within_toolcall_limit",),
        turn_within_fields=("within_toolcall_limit",),
        success_within_fields=("success_within_toolcall_limit",),
    ),
}

SPEC_ORDER = ("token", "turn", "toolcall")


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def iter_turns(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    turns = record.get("turns")
    if not isinstance(turns, list):
        return []
    return [turn for turn in turns if isinstance(turn, dict)]


def first_non_none(mapping: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def find_limit(record: Dict[str, Any], spec: BudgetSpec) -> Optional[int]:
    direct_limit = coerce_int(first_non_none(record, spec.limit_fields))
    if direct_limit is not None:
        return direct_limit

    turn_limits = {
        coerce_int(first_non_none(turn, spec.limit_fields))
        for turn in iter_turns(record)
        if coerce_int(first_non_none(turn, spec.limit_fields)) is not None
    }
    if len(turn_limits) == 1:
        return next(iter(turn_limits))
    return None


def infer_budget_type(record: Dict[str, Any]) -> Optional[str]:
    for budget_type in SPEC_ORDER:
        spec = BUDGET_SPECS[budget_type]
        if find_limit(record, spec) is not None:
            return budget_type

    mode = str(record.get("mode") or "").lower()
    if "token" in mode:
        return "token"
    if "turn" in mode:
        return "turn"
    if "toolcall" in mode or "tool_call" in mode:
        return "toolcall"
    return None


def extract_success(record: Dict[str, Any]) -> Optional[bool]:
    direct = coerce_bool(record.get("success"))
    if direct is not None:
        return direct

    turn_successes = [coerce_bool(turn.get("success")) for turn in iter_turns(record)]
    turn_successes = [value for value in turn_successes if value is not None]
    if turn_successes:
        return turn_successes[-1]

    final_state = str(record.get("final_state") or "").strip().lower()
    if final_state in {"correct", "correct.", "correct!"}:
        return True
    if final_state in {"incorrect", "incorrect.", "incorrect!"}:
        return False
    return None


def compute_over_budget_penalty(limit: int, actual_use: int) -> Optional[float]:
    if limit < 0 or actual_use < 0:
        return None
    if actual_use <= limit:
        return 1.0
    if limit <= 0:
        return 0.0
    return limit / actual_use


def compute_token_within_budget(record: Dict[str, Any], default_limit: int) -> Optional[bool]:
    turns = list(iter_turns(record))
    if turns:
        turn_checks: List[bool] = []
        for turn in turns:
            turn_limit = coerce_int(first_non_none(turn, BUDGET_SPECS["token"].limit_fields))
            if turn_limit is None:
                turn_limit = default_limit

            actual_token = coerce_int(turn.get("actual_token"))
            if actual_token is None:
                actual_token = coerce_int(turn.get("api_total_tokens"))
            if actual_token is None:
                turn_checks = []
                break
            turn_checks.append(actual_token <= turn_limit)

        if turn_checks:
            return all(turn_checks)

    total_tokens = coerce_int(record.get("api_total_tokens"))
    if total_tokens is None:
        total_tokens = 0
        found_any = False
        for turn in turns:
            turn_total = coerce_int(turn.get("api_total_tokens"))
            if turn_total is None:
                turn_total = coerce_int(turn.get("actual_token"))
            if turn_total is None:
                continue
            total_tokens += turn_total
            found_any = True
        if not found_any:
            return None

    return total_tokens <= default_limit


def compute_token_success_score(record: Dict[str, Any], default_limit: int) -> Optional[float]:
    turns = list(iter_turns(record))
    if turns:
        score = 1.0
        for turn in turns:
            turn_limit = coerce_int(first_non_none(turn, BUDGET_SPECS["token"].limit_fields))
            if turn_limit is None:
                turn_limit = default_limit

            actual_token = coerce_int(turn.get("actual_token"))
            if actual_token is None:
                actual_token = coerce_int(turn.get("api_total_tokens"))
            if actual_token is None:
                return None

            penalty = compute_over_budget_penalty(turn_limit, actual_token)
            if penalty is None:
                return None
            score *= penalty
        return score

    total_tokens = coerce_int(record.get("api_total_tokens"))
    if total_tokens is None:
        return None

    return compute_over_budget_penalty(default_limit, total_tokens)


def compute_turn_within_budget(record: Dict[str, Any], default_limit: int) -> Optional[bool]:
    total_turns = coerce_int(record.get("total_turns"))
    if total_turns is None:
        total_turns = len(list(iter_turns(record)))
    if total_turns <= 0:
        return None
    return total_turns <= default_limit


def compute_turn_success_score(record: Dict[str, Any], default_limit: int) -> Optional[float]:
    total_turns = coerce_int(record.get("total_turns"))
    if total_turns is None:
        total_turns = len(list(iter_turns(record)))
    if total_turns <= 0:
        return None
    return compute_over_budget_penalty(default_limit, total_turns)


def compute_toolcall_within_budget(record: Dict[str, Any], default_limit: int) -> Optional[bool]:
    total_toolcalls = coerce_int(record.get("total_toolcalls_used"))
    if total_toolcalls is None:
        total_toolcalls = 0
        found_any = False
        for turn in iter_turns(record):
            actions = turn.get("actions")
            action_names = turn.get("action_names")
            if isinstance(actions, list):
                total_toolcalls += len(actions)
                found_any = True
            elif isinstance(action_names, list):
                total_toolcalls += len(action_names)
                found_any = True
        if not found_any:
            return None
    return total_toolcalls <= default_limit


def compute_toolcall_success_score(record: Dict[str, Any], default_limit: int) -> Optional[float]:
    total_toolcalls = coerce_int(record.get("total_toolcalls_used"))
    if total_toolcalls is None:
        total_toolcalls = 0
        found_any = False
        for turn in iter_turns(record):
            actions = turn.get("actions")
            action_names = turn.get("action_names")
            if isinstance(actions, list):
                total_toolcalls += len(actions)
                found_any = True
            elif isinstance(action_names, list):
                total_toolcalls += len(action_names)
                found_any = True
        if not found_any:
            return None
    return compute_over_budget_penalty(default_limit, total_toolcalls)


def extract_within_budget(record: Dict[str, Any], spec: BudgetSpec, limit: int) -> Optional[bool]:
    direct = coerce_bool(first_non_none(record, spec.row_within_fields))
    if direct is not None:
        return direct

    turn_values = [
        coerce_bool(first_non_none(turn, spec.turn_within_fields))
        for turn in iter_turns(record)
    ]
    turn_values = [value for value in turn_values if value is not None]
    if turn_values:
        return all(turn_values)

    if spec.budget_type == "token":
        return compute_token_within_budget(record, limit)
    if spec.budget_type == "turn":
        return compute_turn_within_budget(record, limit)
    if spec.budget_type == "toolcall":
        return compute_toolcall_within_budget(record, limit)
    return None


def extract_penalized_success_score(
    record: Dict[str, Any], spec: BudgetSpec, limit: int
) -> Optional[float]:
    success = extract_success(record)
    if success is False:
        return 0.0
    if success is None:
        return None

    if spec.budget_type == "token":
        return compute_token_success_score(record, limit)
    if spec.budget_type == "turn":
        return compute_turn_success_score(record, limit)
    if spec.budget_type == "toolcall":
        return compute_toolcall_success_score(record, limit)
    return None


def load_payload(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path.name} does not contain a top-level list.")
    return [item for item in payload if isinstance(item, dict)]


def clean_label(text: str) -> str:
    label = text
    label = label.replace("_eval_compliance_eval_compliance_dialogues", "")
    label = label.replace("_eval_compliance_dialogues", "")
    label = label.replace("-", " ")
    label = re.sub(r"\s+", " ", label).strip()
    return label.replace("_", " ")


def collect_dataset_groups() -> List[Tuple[str, str, List[Path]]]:
    grouped_paths: Dict[Path, List[Path]] = defaultdict(list)
    for path in sorted(DATASET_DIR.rglob(JSON_PATTERN)):
        grouped_paths[path.parent].append(path)

    if grouped_paths:
        groups: List[Tuple[str, str, List[Path]]] = []
        for directory, paths in sorted(grouped_paths.items(), key=lambda item: str(item[0])):
            relative_dir = str(directory.relative_to(DATASET_DIR))
            groups.append((relative_dir, clean_label(directory.name), sorted(paths)))
        return groups

    standalone_paths = sorted(DATASET_DIR.glob("*.json"))
    groups = []
    for path in standalone_paths:
        groups.append((path.name, clean_label(path.stem), [path]))
    return groups


def summarize_group(
    source_dir: str, label: str, json_paths: List[Path]
) -> Tuple[List[Dict[str, Any]], int]:
    aggregate: Dict[Tuple[str, int], Dict[str, Any]] = defaultdict(
        lambda: {
            "source_dir": source_dir,
            "source_json_count": len(json_paths),
            "label": label,
            "budget_type": "",
            "budget_limit": 0,
            "total_samples": 0,
            "raw_success_count": 0,
            "within_budget_count": 0,
            "success_count": 0.0,
        }
    )
    skipped = 0

    for path in json_paths:
        payload = load_payload(path)
        for record in payload:
            budget_type = infer_budget_type(record)
            if budget_type is None:
                skipped += 1
                continue

            spec = BUDGET_SPECS[budget_type]
            limit = find_limit(record, spec)
            if limit is None:
                skipped += 1
                continue

            success = extract_success(record)
            within_budget = extract_within_budget(record, spec, limit)
            success_score = extract_penalized_success_score(record, spec, limit)

            if success_score is None:
                skipped += 1
                continue

            bucket = aggregate[(budget_type, limit)]
            bucket["budget_type"] = budget_type
            bucket["budget_limit"] = limit
            bucket["total_samples"] += 1
            if success is True:
                bucket["raw_success_count"] += 1
            if within_budget is True:
                bucket["within_budget_count"] += 1
            bucket["success_count"] += success_score

    rows: List[Dict[str, Any]] = []
    for budget_type, limit in sorted(aggregate, key=lambda item: (SPEC_ORDER.index(item[0]), item[1])):
        bucket = aggregate[(budget_type, limit)]
        total_samples = bucket["total_samples"]
        bucket["success_rate"] = (
            bucket["success_count"] / total_samples if total_samples else 0.0
        )
        rows.append(bucket)

    return rows, skipped


def write_csv(rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "source_dir",
        "source_json_count",
        "label",
        "budget_type",
        "budget_limit",
        "total_samples",
        "raw_success_count",
        "within_budget_count",
        "success_count",
        "success_rate",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_rows(rows: List[Dict[str, Any]]) -> None:
    present_types = [budget_type for budget_type in SPEC_ORDER if any(row["budget_type"] == budget_type for row in rows)]
    if not present_types:
        raise ValueError("No valid compliance rows were found to plot.")

    fig, axes = plt.subplots(1, len(present_types), figsize=(7.5 * len(present_types), 5.5))
    if len(present_types) == 1:
        axes = [axes]

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["budget_type"], row["label"])].append(row)

    for axis, budget_type in zip(axes, present_types):
        spec = BUDGET_SPECS[budget_type]
        series_labels = sorted({row["label"] for row in rows if row["budget_type"] == budget_type})

        for label in series_labels:
            series = sorted(grouped[(budget_type, label)], key=lambda item: item["budget_limit"])
            x_values = [item["budget_limit"] for item in series]
            y_values = [item["success_rate"] for item in series]
            axis.plot(x_values, y_values, marker="o", linewidth=2, markersize=6, label=label)

        axis.set_title(spec.title)
        axis.set_xlabel(spec.x_label)
        axis.set_ylabel("Penalized Success Rate")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
        axis.set_ylim(0, 1)
        axis.legend(frameon=False)

    fig.suptitle("Penalized Success Rate Under Budget Constraints", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    groups = collect_dataset_groups()
    if not groups:
        raise FileNotFoundError(f"No dataset folders or JSON files found in {DATASET_DIR}")

    all_rows: List[Dict[str, Any]] = []
    total_skipped = 0

    for source_dir, label, json_paths in groups:
        rows, skipped = summarize_group(source_dir, label, json_paths)
        all_rows.extend(rows)
        total_skipped += skipped

    if not all_rows:
        raise ValueError("No rows could be summarized from the compliance dataset files.")

    write_csv(all_rows)
    plot_rows(all_rows)

    print(f"Wrote CSV: {OUTPUT_CSV}")
    print(f"Wrote figure: {OUTPUT_PNG}")
    print(f"Skipped records with unresolved budget/success state: {total_skipped}")
    for row in all_rows:
        print(
            f"{row['label']} | {row['budget_type']}={row['budget_limit']} | "
            f"success={row['success_count']:.4f}/{row['total_samples']} | "
            f"rate={row['success_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
