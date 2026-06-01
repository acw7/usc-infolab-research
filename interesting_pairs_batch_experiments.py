from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import interesting_pairs_batch as ipb


DEFAULT_ROW_BUDGETS = [100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000]
DEFAULT_MIN_PATH_HEXES = [2, 3, 4, 5, 6, 8, 10, 12]
DEFAULT_TIME_THRESHOLDS = [900, 1_800, 3_600, 7_200, 14_400]
DEFAULT_OUTPUT_DIR = Path("interesting_pairs_batch_experiments")


def parse_int_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run timing experiments for interesting_pairs_batch.py and export "
            "raw data plus simple plots."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=ipb.DATA_DIR)
    parser.add_argument(
        "--parquet-file",
        type=Path,
        default=None,
        help="Optional single parquet file to use for all experiments.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help="Number of parquet files to include when --parquet-file is not set.",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=0,
        help="Optional per-file row cap. Use 0 to split total rows across files.",
    )
    parser.add_argument(
        "--row-budgets",
        type=parse_int_list,
        default=DEFAULT_ROW_BUDGETS,
        help="Comma-separated total row budgets for the rows-vs-time sweep.",
    )
    parser.add_argument(
        "--min-path-hexes-values",
        type=parse_int_list,
        default=DEFAULT_MIN_PATH_HEXES,
        help="Comma-separated min-path-hexes values for the hexagons-vs-time sweep.",
    )
    parser.add_argument(
        "--time-threshold-values",
        type=parse_int_list,
        default=DEFAULT_TIME_THRESHOLDS,
        help="Comma-separated time-threshold values in seconds for the threshold sweep.",
    )
    parser.add_argument(
        "--baseline-total-rows",
        type=int,
        default=1_000_000,
        help="Fixed total row budget for non-row sweeps.",
    )
    parser.add_argument(
        "--baseline-min-path-hexes",
        type=int,
        default=ipb.MIN_PATH_HEXES,
        help="Fixed min-path-hexes value for the row and time-threshold sweeps.",
    )
    parser.add_argument(
        "--baseline-time-threshold",
        type=float,
        default=ipb.TIME_THRESHOLD,
        help="Fixed time-threshold in seconds for the row and min-path-hexes sweeps.",
    )
    parser.add_argument(
        "--min-unique-hexes-per-user",
        type=int,
        default=ipb.MIN_UNIQUE_HEXES_PER_USER,
    )
    parser.add_argument(
        "--max-unique-hexes-per-user",
        type=int,
        default=ipb.MAX_UNIQUE_HEXES_PER_USER,
    )
    parser.add_argument("--max-trip-length", type=int, default=ipb.MAX_TRIP_LENGTH)
    parser.add_argument("--min-km-dist", type=float, default=ipb.MIN_KM_DIST)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat each trial and average the runtime.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_parquet_files(args: argparse.Namespace) -> list[Path]:
    return ipb.list_parquet_files(args.data_dir, args.parquet_file, args.max_files)


def run_summary_trial(
    parquet_files: list[Path],
    data_dir: Path,
    total_rows: int,
    rows_per_file: int,
    min_unique_hexes_per_user: int,
    max_unique_hexes_per_user: int,
    time_threshold: float,
    max_trip_length: int,
    min_path_hexes: int,
    min_km_dist: float,
    repeats: int,
) -> dict[str, int | float]:
    if repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    runtimes = []
    final_stats: dict[str, int | float] | None = None

    for _ in range(repeats):
        ipb.reset_runtime_state()
        pair_aggregates: dict[tuple[int, int], ipb.PairAggregate] = defaultdict(ipb.PairAggregate)
        total_sequences = 0
        rows_loaded = 0
        started = time.perf_counter()

        for index, path in enumerate(parquet_files, start=1):
            row_cap = ipb.row_cap_for_file(index, len(parquet_files), total_rows, rows_per_file)
            frame = ipb.read_parquet_subset(
                path,
                nrows=row_cap,
                columns=[ipb.USER_COL, ipb.TIME_COL, ipb.LAT_COL, ipb.LON_COL],
            )
            rows_loaded += len(frame)
            sequences = ipb.build_user_sequences_with_time(
                frame,
                ipb.RES,
                min_unique_hexes_per_user,
                max_unique_hexes_per_user,
            )
            total_sequences += len(sequences)
            ipb.update_pair_aggregates(
                sequences,
                pair_aggregates,
                time_threshold,
                max_trip_length,
                min_path_hexes,
                min_km_dist,
            )

        summary = ipb.build_trip_summary_frame(pair_aggregates)
        elapsed = time.perf_counter() - started
        runtimes.append(elapsed)
        final_stats = {
            "rows_loaded": rows_loaded,
            "sequence_count": total_sequences,
            "pair_count": len(pair_aggregates),
            "summary_rows": len(summary),
            "top_trip_count": int(summary["trip_count"].max()) if not summary.empty else 0,
        }

    assert final_stats is not None
    final_stats["runtime_seconds"] = statistics.mean(runtimes)
    final_stats["runtime_min_seconds"] = min(runtimes)
    final_stats["runtime_max_seconds"] = max(runtimes)
    final_stats["runtime_stddev_seconds"] = statistics.pstdev(runtimes) if len(runtimes) > 1 else 0.0
    final_stats["repeat_count"] = repeats
    return final_stats


def make_result_record(
    experiment_type: str,
    parquet_files: list[Path],
    total_rows: int,
    rows_per_file: int,
    time_threshold: float,
    min_path_hexes: int,
    trial_stats: dict[str, int | float],
) -> dict[str, int | float | str]:
    return {
        "experiment_type": experiment_type,
        "file_count": len(parquet_files),
        "files": "|".join(str(path) for path in parquet_files),
        "total_rows_budget": total_rows,
        "rows_per_file": rows_per_file,
        "time_threshold_seconds": time_threshold,
        "min_path_hexes": min_path_hexes,
        "rows_loaded": int(trial_stats["rows_loaded"]),
        "sequence_count": int(trial_stats["sequence_count"]),
        "pair_count": int(trial_stats["pair_count"]),
        "summary_rows": int(trial_stats["summary_rows"]),
        "top_trip_count": int(trial_stats["top_trip_count"]),
        "repeat_count": int(trial_stats["repeat_count"]),
        "runtime_seconds": round(float(trial_stats["runtime_seconds"]), 6),
        "runtime_stddev_seconds": round(float(trial_stats["runtime_stddev_seconds"]), 6),
        "runtime_min_seconds": round(float(trial_stats["runtime_min_seconds"]), 6),
        "runtime_max_seconds": round(float(trial_stats["runtime_max_seconds"]), 6),
    }


def write_csv(records: list[dict[str, int | float | str]], out_path: Path) -> None:
    if not records:
        raise ValueError("No experiment records were generated.")

    excluded_columns = {"files", "rows_per_file"}
    fieldnames = [key for key in records[0].keys() if key not in excluded_columns]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: value for key, value in record.items() if key in fieldnames})


def write_json(payload: dict[str, object], out_path: Path) -> None:
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def nice_tick_values(min_value: float, max_value: float, tick_count: int = 5) -> list[float]:
    if min_value == max_value:
        return [min_value]

    span = max_value - min_value
    raw_step = span / max(tick_count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    candidates = [1, 2, 5, 10]
    step = candidates[-1] * magnitude
    for candidate in candidates:
        candidate_step = candidate * magnitude
        if raw_step <= candidate_step:
            step = candidate_step
            break

    first_tick = math.floor(min_value / step) * step
    last_tick = math.ceil(max_value / step) * step
    ticks = []
    current = first_tick
    while current <= last_tick + (step * 0.5):
        ticks.append(round(current, 10))
        current += step
    return ticks


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_line_plot_svg(
    records: list[dict[str, int | float | str]],
    x_key: str,
    x_label: str,
    title: str,
    out_path: Path,
    *,
    x_log_scale: bool = False,
) -> None:
    if not records:
        return

    width = 900
    height = 520
    margin_left = 90
    margin_right = 30
    margin_top = 70
    margin_bottom = 80
    inner_width = width - margin_left - margin_right
    inner_height = height - margin_top - margin_bottom

    x_values = [float(record[x_key]) for record in records]
    y_values = [float(record["runtime_seconds"]) for record in records]

    if x_log_scale:
        if min(x_values) <= 0:
            raise ValueError("Log-scaled plots require strictly positive x values.")
        scaled_x_values = [math.log10(value) for value in x_values]
    else:
        scaled_x_values = x_values

    min_x = min(scaled_x_values)
    max_x = max(scaled_x_values)
    min_y = 0.0
    max_y = max(y_values)
    if min_x == max_x:
        min_x -= 0.5
        max_x += 0.5
    if max_y == min_y:
        max_y += 1.0
    else:
        max_y *= 1.1

    def project_x(value: float) -> float:
        scaled = math.log10(value) if x_log_scale else value
        return margin_left + ((scaled - min_x) / (max_x - min_x)) * inner_width

    def project_y(value: float) -> float:
        return margin_top + inner_height - ((value - min_y) / (max_y - min_y)) * inner_height

    x_ticks = x_values if len(x_values) <= 8 else nice_tick_values(min(x_values), max(x_values))
    if x_log_scale:
        x_ticks = [tick for tick in x_ticks if tick > 0]
    y_ticks = [
        tick
        for tick in nice_tick_values(min_y, max_y)
        if (min_y - 1e-9) <= tick <= (max_y + 1e-9)
    ]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f9fafb" />',
        f'<text x="{width / 2:.1f}" y="36" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="22" fill="#111827">{svg_escape(title)}</text>',
        f'<text x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#374151">{svg_escape(x_label)}</text>',
        (
            f'<text x="28" y="{height / 2:.1f}" text-anchor="middle" '
            'font-family="Helvetica, Arial, sans-serif" font-size="16" '
            'fill="#374151" transform="rotate(-90 28 '
            f'{height / 2:.1f})">Runtime (seconds)</text>'
        ),
    ]

    for tick in y_ticks:
        y = project_y(float(tick))
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />'
        )
        lines.append(
            f'<text x="{margin_left - 10}" y="{y + 5:.2f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#4b5563">{svg_escape(format_seconds(float(tick)))}</text>'
        )

    for tick in x_ticks:
        x = project_x(float(tick))
        label = f"{int(tick):,}" if float(tick).is_integer() else f"{tick:g}"
        lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#eef2f7" stroke-width="1" />'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#4b5563">{svg_escape(label)}</text>'
        )

    lines.append(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#111827" stroke-width="1.5" />'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#111827" stroke-width="1.5" />'
    )

    points = [
        (project_x(float(record[x_key])), project_y(float(record["runtime_seconds"])), record)
        for record in records
    ]
    polyline_points = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
    lines.append(
        f'<polyline fill="none" stroke="#0f766e" stroke-width="3" points="{polyline_points}" />'
    )

    for x, y, record in points:
        runtime_text = format_seconds(float(record["runtime_seconds"]))
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#0f766e" />')
        lines.append(
            f'<text x="{x:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#134e4a">{svg_escape(runtime_text)}</text>'
        )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n")


def build_summary_payload(
    args: argparse.Namespace,
    parquet_files: list[Path],
    records: list[dict[str, int | float | str]],
) -> dict[str, object]:
    return {
        "generated_at_unix_seconds": time.time(),
        "data_dir": str(args.data_dir),
        "parquet_files": [str(path) for path in parquet_files],
        "parameters": {
            "rows_per_file": args.rows_per_file,
            "row_budgets": args.row_budgets,
            "min_path_hexes_values": args.min_path_hexes_values,
            "time_threshold_values": args.time_threshold_values,
            "baseline_total_rows": args.baseline_total_rows,
            "baseline_min_path_hexes": args.baseline_min_path_hexes,
            "baseline_time_threshold": args.baseline_time_threshold,
            "min_unique_hexes_per_user": args.min_unique_hexes_per_user,
            "max_unique_hexes_per_user": args.max_unique_hexes_per_user,
            "max_trip_length": args.max_trip_length,
            "min_km_dist": args.min_km_dist,
            "repeats": args.repeats,
        },
        "results": records,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = resolve_parquet_files(args)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {args.data_dir.resolve()}")

    print("Using parquet files:")
    for path in parquet_files:
        print(f"  - {path}")

    records: list[dict[str, int | float | str]] = []

    print("\nRunning rows-vs-time sweep...")
    for total_rows in args.row_budgets:
        trial_stats = run_summary_trial(
            parquet_files,
            args.data_dir,
            total_rows,
            args.rows_per_file,
            args.min_unique_hexes_per_user,
            args.max_unique_hexes_per_user,
            args.baseline_time_threshold,
            args.max_trip_length,
            args.baseline_min_path_hexes,
            args.min_km_dist,
            args.repeats,
        )
        record = make_result_record(
            "rows_vs_time",
            parquet_files,
            total_rows,
            args.rows_per_file,
            args.baseline_time_threshold,
            args.baseline_min_path_hexes,
            trial_stats,
        )
        records.append(record)
        print(
            f"  total_rows={total_rows:,} -> runtime={record['runtime_seconds']:.2f}s, "
            f"pairs={record['pair_count']:,}, sequences={record['sequence_count']:,}"
        )

    print("\nRunning min-path-hexes-vs-time sweep...")
    for min_path_hexes in args.min_path_hexes_values:
        trial_stats = run_summary_trial(
            parquet_files,
            args.data_dir,
            args.baseline_total_rows,
            args.rows_per_file,
            args.min_unique_hexes_per_user,
            args.max_unique_hexes_per_user,
            args.baseline_time_threshold,
            args.max_trip_length,
            min_path_hexes,
            args.min_km_dist,
            args.repeats,
        )
        record = make_result_record(
            "min_path_hexes_vs_time",
            parquet_files,
            args.baseline_total_rows,
            args.rows_per_file,
            args.baseline_time_threshold,
            min_path_hexes,
            trial_stats,
        )
        records.append(record)
        print(
            f"  min_path_hexes={min_path_hexes} -> runtime={record['runtime_seconds']:.2f}s, "
            f"pairs={record['pair_count']:,}, sequences={record['sequence_count']:,}"
        )

    print("\nRunning time-threshold-vs-time sweep...")
    for time_threshold in args.time_threshold_values:
        trial_stats = run_summary_trial(
            parquet_files,
            args.data_dir,
            args.baseline_total_rows,
            args.rows_per_file,
            args.min_unique_hexes_per_user,
            args.max_unique_hexes_per_user,
            time_threshold,
            args.max_trip_length,
            args.baseline_min_path_hexes,
            args.min_km_dist,
            args.repeats,
        )
        record = make_result_record(
            "time_threshold_vs_time",
            parquet_files,
            args.baseline_total_rows,
            args.rows_per_file,
            time_threshold,
            args.baseline_min_path_hexes,
            trial_stats,
        )
        records.append(record)
        print(
            f"  time_threshold={int(time_threshold):,}s -> runtime={record['runtime_seconds']:.2f}s, "
            f"pairs={record['pair_count']:,}, sequences={record['sequence_count']:,}"
        )

    csv_path = args.output_dir / "experiment_results.csv"
    json_path = args.output_dir / "experiment_results.json"
    rows_svg_path = args.output_dir / "rows_vs_time.svg"
    hexes_svg_path = args.output_dir / "min_path_hexes_vs_time.svg"
    threshold_svg_path = args.output_dir / "time_threshold_vs_time.svg"

    write_csv(records, csv_path)
    write_json(build_summary_payload(args, parquet_files, records), json_path)

    rows_records = [record for record in records if record["experiment_type"] == "rows_vs_time"]
    hex_records = [
        record for record in records if record["experiment_type"] == "min_path_hexes_vs_time"
    ]
    threshold_records = [
        record for record in records if record["experiment_type"] == "time_threshold_vs_time"
    ]

    write_line_plot_svg(
        rows_records,
        "total_rows_budget",
        "Total rows budget",
        "interesting_pairs_batch: Total Rows vs Runtime",
        rows_svg_path,
        x_log_scale=True,
    )
    write_line_plot_svg(
        hex_records,
        "min_path_hexes",
        "Minimum path hexagons",
        "interesting_pairs_batch: Minimum Path Hexagons vs Runtime",
        hexes_svg_path,
    )
    write_line_plot_svg(
        threshold_records,
        "time_threshold_seconds",
        "Time threshold (seconds)",
        "interesting_pairs_batch: Time Threshold vs Runtime",
        threshold_svg_path,
    )

    print("\nSaved outputs:")
    print(f"  - {csv_path.resolve()}")
    print(f"  - {json_path.resolve()}")
    print(f"  - {rows_svg_path.resolve()}")
    print(f"  - {hexes_svg_path.resolve()}")
    print(f"  - {threshold_svg_path.resolve()}")


if __name__ == "__main__":
    main()
