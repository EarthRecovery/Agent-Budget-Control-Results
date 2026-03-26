#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
INPUT_GLOB = "*.json"

BENCHMARK_LABELS = {
    "CoordSokoban": "sokoban",
    "WebShop": "webshop",
    "CoordFrozenLake": "frozen-lake",
    "DeepCoder": "deepcoder",
    "SearchQA": "search-r1",
    "GPQAMain": "gpqa-main",
}

BENCHMARK_ORDER = [
    "sokoban",
    "webshop",
    "frozen-lake",
    "deepcoder",
    "search-r1",
    "gpqa-main",
]

MODEL_ORDER = [
    "Qwen32B-Think",
    "Qwen32B-Instant",
    "GPT5.2-Think",
    "GPT5.2-Instant",
]


def coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compute_accuracy_factor(estimate: Optional[int], actual: Optional[int]) -> Optional[float]:
    if estimate is None or actual is None:
        return None
    actual_value = max(1, int(actual))
    estimate_value = max(0, int(estimate))
    error_ratio = abs(estimate_value - actual_value) / float(actual_value)
    return max(0.0, 1.0 - error_ratio)


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def extract_total_tokens(record: Dict[str, Any]) -> Optional[int]:
    total_tokens = coerce_optional_int(record.get("api_total_tokens"))
    if total_tokens is not None:
        return total_tokens

    input_tokens = coerce_optional_int(record.get("api_input_tokens"))
    output_tokens = coerce_optional_int(record.get("api_output_tokens"))
    if input_tokens is None or output_tokens is None:
        return None
    return input_tokens + output_tokens


def compute_total_used_tokens(payload: Any) -> Optional[int]:
    if not isinstance(payload, list):
        return None

    total_used_tokens = 0
    found_any = False
    for env_record in payload:
        if not isinstance(env_record, dict):
            continue

        env_total_tokens = extract_total_tokens(env_record)
        if env_total_tokens is not None:
            total_used_tokens += env_total_tokens
            found_any = True
            continue

        turns = env_record.get("turns")
        if not isinstance(turns, list):
            continue

        turn_total_tokens = 0
        found_turn_tokens = False
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            turn_total = extract_total_tokens(turn)
            if turn_total is None:
                continue
            turn_total_tokens += turn_total
            found_turn_tokens = True

        if found_turn_tokens:
            total_used_tokens += turn_total_tokens
            found_any = True

    if not found_any:
        return None
    return total_used_tokens


def compute_env_used_tokens(env_record: Dict[str, Any]) -> Optional[int]:
    env_total_tokens = extract_total_tokens(env_record)
    if env_total_tokens is not None:
        return env_total_tokens

    turns = env_record.get("turns")
    if not isinstance(turns, list):
        return None

    total_tokens = 0
    found_any = False
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        turn_total = extract_total_tokens(turn)
        if turn_total is None:
            continue
        total_tokens += turn_total
        found_any = True

    if not found_any:
        return None
    return total_tokens


def iter_turns(payload: Any) -> Iterable[Dict[str, Any]]:
    if not isinstance(payload, list):
        return
    for env_record in payload:
        if not isinstance(env_record, dict):
            continue
        turns = env_record.get("turns")
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            yield turn


def detect_model(path: Path) -> str:
    name = path.name.lower()
    if "qwen" in name and "thinking" in name:
        return "Qwen32B-Think"
    if "qwen" in name and "instant" in name:
        return "Qwen32B-Instant"
    if "gpt5.2thinking" in name:
        return "GPT5.2-Think"
    if "gpt5.2instant" in name:
        return "GPT5.2-Instant"
    raise ValueError(f"Unable to detect model from filename: {path.name}")


def detect_benchmark(payload: Any, path: Path) -> str:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        tag = payload[0].get("tag")
        if tag in BENCHMARK_LABELS:
            return BENCHMARK_LABELS[tag]

    stem = path.stem.lower()
    for key in BENCHMARK_ORDER:
        if key in stem or key.replace("-", "_") in stem:
            return key
    raise ValueError(f"Unable to detect benchmark for file: {path.name}")


def summarize_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    benchmark = detect_benchmark(payload, path)
    model = detect_model(path)
    mode = None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        mode = payload[0].get("mode")
    multi_turn = mode == "multi"

    token_scores: List[float] = []
    turn_scores: List[float] = []
    total_turns = 0
    token_valid_turns = 0
    turn_valid_turns = 0

    for turn in iter_turns(payload):
        total_turns += 1
        generation_error = turn.get("generation_error")
        generation_success = turn.get("generation_success")
        if generation_error is not None or generation_success is False:
            continue

        token_score = compute_accuracy_factor(
            coerce_optional_int(turn.get("estimate_token")),
            coerce_optional_int(turn.get("actual_token")),
        )
        if token_score is not None:
            token_scores.append(token_score)
            token_valid_turns += 1

        if multi_turn:
            turn_score = compute_accuracy_factor(
                coerce_optional_int(turn.get("estimate_remaining_turn")),
                coerce_optional_int(turn.get("actual_remaining_turn")),
            )
            if turn_score is not None:
                turn_scores.append(turn_score)
                turn_valid_turns += 1

    return {
        "file": path.name,
        "benchmark": benchmark,
        "model": model,
        "multi_turn": multi_turn,
        "mode": mode,
        "total_turns": total_turns,
        "token_valid_turns": token_valid_turns,
        "turn_valid_turns": turn_valid_turns,
        "token_accuracy": mean_or_none(token_scores),
        "turn_accuracy": mean_or_none(turn_scores),
    }


def iter_unique_latest_env_records(payload: Any) -> Iterable[Dict[str, Any]]:
    if not isinstance(payload, list):
        return

    latest_by_env: Dict[Any, Tuple[Tuple[int, int], Dict[str, Any]]] = {}
    for index, env_record in enumerate(payload):
        if not isinstance(env_record, dict):
            continue

        env_id = env_record.get("env_id")
        env_key = ("env_id", env_id) if env_id is not None else ("row_index", index)
        absolute_env_id = coerce_optional_int(env_record.get("absolute_env_id"))
        rank = (absolute_env_id if absolute_env_id is not None else index, index)

        previous = latest_by_env.get(env_key)
        if previous is None or rank >= previous[0]:
            latest_by_env[env_key] = (rank, env_record)

    for _, env_record in sorted(latest_by_env.values(), key=lambda item: item[0]):
        yield env_record


def env_has_generation_error(env_record: Dict[str, Any]) -> bool:
    turns = env_record.get("turns")
    if not isinstance(turns, list) or not turns:
        return True

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        if turn.get("generation_error") is not None or turn.get("generation_success") is False:
            return True
    return False


def env_success(env_record: Dict[str, Any]) -> bool:
    turns = env_record.get("turns")
    if not isinstance(turns, list) or not turns:
        return False

    last_turn = turns[-1]
    if not isinstance(last_turn, dict):
        return False
    return last_turn.get("success") is True


def env_reward_positive(env_record: Dict[str, Any]) -> bool:
    turns = env_record.get("turns")
    if not isinstance(turns, list) or not turns:
        return False

    last_turn = turns[-1]
    if not isinstance(last_turn, dict):
        return False

    reward = last_turn.get("reward")
    try:
        return float(reward) > 0.0
    except (TypeError, ValueError):
        return False


def summarize_run_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    benchmark = detect_benchmark(payload, path)
    model = detect_model(path)
    mode = None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        mode = payload[0].get("mode")
    multi_turn = mode == "multi"

    unique_envs = list(iter_unique_latest_env_records(payload))
    unique_env_count = len(unique_envs)
    completed_envs = sum(1 for env_record in unique_envs if not env_has_generation_error(env_record))
    incomplete_envs = unique_env_count - completed_envs
    success_count = sum(1 for env_record in unique_envs if env_success(env_record))
    reward_positive_count = sum(1 for env_record in unique_envs if env_reward_positive(env_record))

    completion_ratio: Optional[float]
    if unique_env_count == 0:
        completion_ratio = None
    elif incomplete_envs == 0:
        completion_ratio = math.inf
    else:
        completion_ratio = completed_envs / float(incomplete_envs)

    return {
        "file": path.name,
        "benchmark": benchmark,
        "model": model,
        "multi_turn": multi_turn,
        "mode": mode,
        "raw_records": len(payload) if isinstance(payload, list) else 0,
        "unique_envs": unique_env_count,
        "success_count": success_count,
        "success_rate": (success_count / float(unique_env_count)) if unique_env_count else None,
        "reward_positive_count": reward_positive_count,
        "reward_positive_rate": (
            reward_positive_count / float(unique_env_count) if unique_env_count else None
        ),
        "completed_envs": completed_envs,
        "incomplete_envs": incomplete_envs,
        "completion_ratio": completion_ratio,
        "total_used_tokens": compute_total_used_tokens(payload),
    }


def mean_float_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def percentile_or_none(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    if lower_index == upper_index:
        return lower_value

    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def env_turn_count(env_record: Dict[str, Any]) -> Optional[int]:
    turns = env_record.get("turns")
    if isinstance(turns, list):
        return len(turns)
    return coerce_optional_int(env_record.get("total_turns"))


def summarize_usage_stats(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    benchmark = detect_benchmark(payload, path)
    model = detect_model(path)
    mode = None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        mode = payload[0].get("mode")
    multi_turn = mode == "multi"

    unique_envs = list(iter_unique_latest_env_records(payload))
    token_values = [
        float(token_total)
        for env_record in unique_envs
        for token_total in [compute_env_used_tokens(env_record)]
        if token_total is not None
    ]
    turn_values = [
        float(turn_count)
        for env_record in unique_envs
        for turn_count in [env_turn_count(env_record)]
        if turn_count is not None
    ]

    return {
        "file": path.name,
        "benchmark": benchmark,
        "model": model,
        "multi_turn": multi_turn,
        "mode": mode,
        "unique_envs": len(unique_envs),
        "token_observed_envs": len(token_values),
        "turn_observed_envs": len(turn_values),
        "mean_token": mean_float_or_none(token_values),
        "p70_token": percentile_or_none(token_values, 0.70),
        "p30_token": percentile_or_none(token_values, 0.30),
        "mean_turn": mean_float_or_none(turn_values),
        "p70_turn": percentile_or_none(turn_values, 0.70),
        "p30_turn": percentile_or_none(turn_values, 0.30),
    }


def ordered_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    benchmark_rank = {name: idx for idx, name in enumerate(BENCHMARK_ORDER)}
    model_rank = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            benchmark_rank.get(row["benchmark"], math.inf),
            model_rank.get(row["model"], math.inf),
        ),
    )


def format_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100.0:.2f}%"


def format_tokens(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000.0:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000.0:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000.0:.2f}K"
    return str(value)


def format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}x"


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def cell_text(row: Dict[str, Any], *, multiline: bool) -> str:
    if row["multi_turn"]:
        sep = "\n" if multiline else " | "
        return (
            f"turn {format_pct(row['turn_accuracy'])}"
            f"{sep}token {format_pct(row['token_accuracy'])}"
        )
    return format_pct(row["token_accuracy"])


def build_table(rows: List[Dict[str, Any]], *, multiline: bool) -> Tuple[List[str], List[List[str]]]:
    headers = ["benchmark", *MODEL_ORDER, "multi-turn"]
    by_key = {(row["benchmark"], row["model"]): row for row in rows}
    table_rows: List[List[str]] = []
    for benchmark in BENCHMARK_ORDER:
        benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
        if not benchmark_rows:
            continue
        multi_turn = "yes" if benchmark_rows[0]["multi_turn"] else "no"
        values = [benchmark]
        for model in MODEL_ORDER:
            row = by_key.get((benchmark, model))
            values.append(cell_text(row, multiline=multiline) if row else "N/A")
        values.append(multi_turn)
        table_rows.append(values)
    return headers, table_rows


def build_run_metrics_table(rows: List[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    headers = [
        "benchmark",
        "model",
        "total tokens",
        "success rate",
        "reward>0 rate",
        "complete",
        "complete ratio",
        "multi-turn",
    ]
    table_rows: List[List[str]] = []
    for row in rows:
        table_rows.append(
            [
                row["benchmark"],
                row["model"],
                format_tokens(row["total_used_tokens"]),
                format_pct(row["success_rate"]),
                format_pct(row["reward_positive_rate"]),
                f"{row['completed_envs']}/{row['incomplete_envs']}",
                format_ratio(row["completion_ratio"]),
                "yes" if row["multi_turn"] else "no",
            ]
        )
    return headers, table_rows


def write_long_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "benchmark",
        "model",
        "multi_turn",
        "mode",
        "token_accuracy",
        "turn_accuracy",
        "token_valid_turns",
        "turn_valid_turns",
        "total_turns",
        "file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "benchmark": row["benchmark"],
                    "model": row["model"],
                    "multi_turn": row["multi_turn"],
                    "mode": row["mode"],
                    "token_accuracy": f"{row['token_accuracy']:.6f}" if row["token_accuracy"] is not None else "",
                    "turn_accuracy": f"{row['turn_accuracy']:.6f}" if row["turn_accuracy"] is not None else "",
                    "token_valid_turns": row["token_valid_turns"],
                    "turn_valid_turns": row["turn_valid_turns"],
                    "total_turns": row["total_turns"],
                    "file": row["file"],
                }
            )


def write_run_metrics_long_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "benchmark",
        "model",
        "multi_turn",
        "mode",
        "raw_records",
        "unique_envs",
        "success_count",
        "success_rate",
        "reward_positive_count",
        "reward_positive_rate",
        "completed_envs",
        "incomplete_envs",
        "completion_ratio",
        "total_used_tokens",
        "file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "benchmark": row["benchmark"],
                    "model": row["model"],
                    "multi_turn": row["multi_turn"],
                    "mode": row["mode"],
                    "raw_records": row["raw_records"],
                    "unique_envs": row["unique_envs"],
                    "success_count": row["success_count"],
                    "success_rate": f"{row['success_rate']:.6f}" if row["success_rate"] is not None else "",
                    "reward_positive_count": row["reward_positive_count"],
                    "reward_positive_rate": (
                        f"{row['reward_positive_rate']:.6f}"
                        if row["reward_positive_rate"] is not None
                        else ""
                    ),
                    "completed_envs": row["completed_envs"],
                    "incomplete_envs": row["incomplete_envs"],
                    "completion_ratio": (
                        "inf"
                        if row["completion_ratio"] is not None and math.isinf(row["completion_ratio"])
                        else (f"{row['completion_ratio']:.6f}" if row["completion_ratio"] is not None else "")
                    ),
                    "total_used_tokens": row["total_used_tokens"] if row["total_used_tokens"] is not None else "",
                    "file": row["file"],
                }
            )


def build_usage_stats_table(rows: List[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    headers = [
        "benchmark",
        "model",
        "mean token",
        "p70 token",
        "p30 token",
        "mean turn",
        "p70 turn",
        "p30 turn",
        "multi-turn",
    ]
    table_rows: List[List[str]] = []
    for row in rows:
        table_rows.append(
            [
                row["benchmark"],
                row["model"],
                format_float(row["mean_token"]),
                format_float(row["p70_token"]),
                format_float(row["p30_token"]),
                format_float(row["mean_turn"]),
                format_float(row["p70_turn"]),
                format_float(row["p30_turn"]),
                "yes" if row["multi_turn"] else "no",
            ]
        )
    return headers, table_rows


def write_usage_stats_long_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "benchmark",
        "model",
        "multi_turn",
        "mode",
        "unique_envs",
        "token_observed_envs",
        "turn_observed_envs",
        "mean_token",
        "p70_token",
        "p30_token",
        "mean_turn",
        "p70_turn",
        "p30_turn",
        "file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "benchmark": row["benchmark"],
                    "model": row["model"],
                    "multi_turn": row["multi_turn"],
                    "mode": row["mode"],
                    "unique_envs": row["unique_envs"],
                    "token_observed_envs": row["token_observed_envs"],
                    "turn_observed_envs": row["turn_observed_envs"],
                    "mean_token": f"{row['mean_token']:.6f}" if row["mean_token"] is not None else "",
                    "p70_token": f"{row['p70_token']:.6f}" if row["p70_token"] is not None else "",
                    "p30_token": f"{row['p30_token']:.6f}" if row["p30_token"] is not None else "",
                    "mean_turn": f"{row['mean_turn']:.6f}" if row["mean_turn"] is not None else "",
                    "p70_turn": f"{row['p70_turn']:.6f}" if row["p70_turn"] is not None else "",
                    "p30_turn": f"{row['p30_turn']:.6f}" if row["p30_turn"] is not None else "",
                    "file": row["file"],
                }
            )


def write_table_csv(headers: List[str], table_rows: List[List[str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(table_rows)


def write_markdown(headers: List[str], table_rows: List[List[str]], path: Path, note_text: str = "") -> None:
    def normalize_cell(value: str) -> str:
        return value.replace("\n", "<br>").replace(" | ", "<br>")

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    content_lines = [header_line, separator_line]
    for row in table_rows:
        content_lines.append("| " + " | ".join(normalize_cell(value) for value in row) + " |")
    if note_text:
        content_lines.append("")
        content_lines.extend(note_text.splitlines())
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(content_lines))


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates: List[str] = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            ]
        )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    lines: List[str] = []
    for raw_line in text.splitlines() or [""]:
        words = raw_line.split(" ")
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines or [""]


def measure_block(draw: ImageDraw.ImageDraw, lines: Sequence[str], font: ImageFont.ImageFont, line_gap: int) -> Tuple[int, int]:
    widths = []
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line or " ", font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    width = max(widths) if widths else 0
    height = sum(heights) + line_gap * max(0, len(heights) - 1)
    return width, height


def draw_table_png(headers: List[str], table_rows: List[List[str]], path: Path, note_text: str = "") -> None:
    header_font = load_font(26, bold=True)
    cell_font = load_font(22, bold=False)
    note_font = load_font(18, bold=False)
    line_gap = 6
    cell_padding_x = 16
    cell_padding_y = 14
    note_lines = note_text.splitlines() if note_text else []

    dummy = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(dummy)

    columns = list(zip(headers, *table_rows))
    col_widths: List[int] = []
    wrapped_cells: List[List[List[str]]] = []

    for col_idx, column in enumerate(columns):
        font = header_font if col_idx == 0 else header_font
        header_lines = wrap_text(draw, column[0], font, max_width=300)
        cell_lines_for_column = [header_lines]
        max_width = measure_block(draw, header_lines, font, line_gap)[0]
        for value in column[1:]:
            wrapped = wrap_text(draw, value, cell_font, max_width=320)
            cell_lines_for_column.append(wrapped)
            max_width = max(max_width, measure_block(draw, wrapped, cell_font, line_gap)[0])
        col_widths.append(max_width + cell_padding_x * 2)
        wrapped_cells.append(cell_lines_for_column)

    row_heights: List[int] = []
    row_count = len(table_rows) + 1
    for row_idx in range(row_count):
        row_height = 0
        for col_idx in range(len(headers)):
            font = header_font if row_idx == 0 else cell_font
            _, block_height = measure_block(draw, wrapped_cells[col_idx][row_idx], font, line_gap)
            row_height = max(row_height, block_height + cell_padding_y * 2)
        row_heights.append(row_height)

    table_width = sum(col_widths)
    table_height = sum(row_heights)
    _, note_block_height = measure_block(draw, note_lines or [""], note_font, line_gap)
    note_height = note_block_height + 24 if note_lines else 0

    margin = 24
    image = Image.new("RGB", (table_width + margin * 2, table_height + note_height + margin * 2), "#ffffff")
    draw = ImageDraw.Draw(image)

    border_color = "#cfcfcf"
    header_fill = "#f2f2f2"
    row_fill_a = "#ffffff"
    row_fill_b = "#fafafa"
    text_color = "#111111"

    y = margin
    for row_idx in range(row_count):
        x = margin
        row_fill = header_fill if row_idx == 0 else (row_fill_a if row_idx % 2 == 1 else row_fill_b)
        for col_idx, width in enumerate(col_widths):
            draw.rectangle(
                [(x, y), (x + width, y + row_heights[row_idx])],
                fill=row_fill,
                outline=border_color,
                width=1,
            )
            lines = wrapped_cells[col_idx][row_idx]
            font = header_font if row_idx == 0 else cell_font
            _, block_height = measure_block(draw, lines, font, line_gap)
            text_y = y + (row_heights[row_idx] - block_height) / 2
            for line in lines:
                bbox = draw.textbbox((0, 0), line or " ", font=font)
                line_height = bbox[3] - bbox[1]
                draw.text((x + cell_padding_x, text_y), line, fill=text_color, font=font)
                text_y += line_height + line_gap
            x += width
        y += row_heights[row_idx]

    if note_lines:
        note_y = margin + table_height + 18
        for line in note_lines:
            bbox = draw.textbbox((0, 0), line or " ", font=note_font)
            line_height = bbox[3] - bbox[1]
            draw.text((margin, note_y), line, fill="#444444", font=note_font)
            note_y += line_height + line_gap
    image.save(path)


def main() -> int:
    json_paths = sorted(ROOT.glob(INPUT_GLOB))
    accuracy_rows = ordered_rows([summarize_file(path) for path in json_paths])
    run_metric_rows = ordered_rows([summarize_run_metrics(path) for path in json_paths])
    usage_stat_rows = ordered_rows([summarize_usage_stats(path) for path in json_paths])

    accuracy_note = "Accuracy = mean(max(0, 1 - |estimate - actual| / max(1, actual))) over valid turns."
    run_metrics_note = (
        "Tokens = total observed API tokens across all records; N/A means token fields are missing.\n"
        "Success rate = successful final outcomes / unique env_id. Reward>0 rate = final-turn reward > 0 / unique env_id.\n"
        "Complete = latest completed envs / incomplete envs, where incomplete means the latest trajectory for that env_id"
        " contains a generation error."
    )
    usage_stats_note = (
        "Each row is computed over the latest trajectory for each unique env_id.\n"
        "p70 / p30 mean the 70th / 30th percentile of per-env token usage or turn count."
    )

    accuracy_long_csv_path = ROOT / "estimation_accuracy_summary_long.csv"
    accuracy_table_csv_path = ROOT / "estimation_accuracy_table.csv"
    accuracy_markdown_path = ROOT / "estimation_accuracy_table.md"
    accuracy_png_path = ROOT / "estimation_accuracy_table.png"

    accuracy_headers_csv, accuracy_table_rows_csv = build_table(accuracy_rows, multiline=False)
    accuracy_headers_png, accuracy_table_rows_png = build_table(accuracy_rows, multiline=True)

    write_long_csv(accuracy_rows, accuracy_long_csv_path)
    write_table_csv(accuracy_headers_csv, accuracy_table_rows_csv, accuracy_table_csv_path)
    write_markdown(accuracy_headers_csv, accuracy_table_rows_csv, accuracy_markdown_path, accuracy_note)
    draw_table_png(accuracy_headers_png, accuracy_table_rows_png, accuracy_png_path, accuracy_note)

    run_metrics_long_csv_path = ROOT / "experiment_run_metrics_long.csv"
    run_metrics_table_csv_path = ROOT / "experiment_run_metrics_table.csv"
    run_metrics_markdown_path = ROOT / "experiment_run_metrics_table.md"
    run_metrics_png_path = ROOT / "experiment_run_metrics_table.png"

    run_headers_csv, run_table_rows_csv = build_run_metrics_table(run_metric_rows)
    run_headers_png, run_table_rows_png = build_run_metrics_table(run_metric_rows)

    write_run_metrics_long_csv(run_metric_rows, run_metrics_long_csv_path)
    write_table_csv(run_headers_csv, run_table_rows_csv, run_metrics_table_csv_path)
    write_markdown(run_headers_csv, run_table_rows_csv, run_metrics_markdown_path, run_metrics_note)
    draw_table_png(run_headers_png, run_table_rows_png, run_metrics_png_path, run_metrics_note)

    usage_stats_long_csv_path = ROOT / "experiment_usage_stats_long.csv"
    usage_stats_table_csv_path = ROOT / "experiment_usage_stats_table.csv"
    usage_stats_markdown_path = ROOT / "experiment_usage_stats_table.md"
    usage_stats_png_path = ROOT / "experiment_usage_stats_table.png"

    usage_headers_csv, usage_table_rows_csv = build_usage_stats_table(usage_stat_rows)
    usage_headers_png, usage_table_rows_png = build_usage_stats_table(usage_stat_rows)

    write_usage_stats_long_csv(usage_stat_rows, usage_stats_long_csv_path)
    write_table_csv(usage_headers_csv, usage_table_rows_csv, usage_stats_table_csv_path)
    write_markdown(usage_headers_csv, usage_table_rows_csv, usage_stats_markdown_path, usage_stats_note)
    draw_table_png(usage_headers_png, usage_table_rows_png, usage_stats_png_path, usage_stats_note)

    print(f"Wrote: {accuracy_long_csv_path}")
    print(f"Wrote: {accuracy_table_csv_path}")
    print(f"Wrote: {accuracy_markdown_path}")
    print(f"Wrote: {accuracy_png_path}")
    print(f"Wrote: {run_metrics_long_csv_path}")
    print(f"Wrote: {run_metrics_table_csv_path}")
    print(f"Wrote: {run_metrics_markdown_path}")
    print(f"Wrote: {run_metrics_png_path}")
    print(f"Wrote: {usage_stats_long_csv_path}")
    print(f"Wrote: {usage_stats_table_csv_path}")
    print(f"Wrote: {usage_stats_markdown_path}")
    print(f"Wrote: {usage_stats_png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
