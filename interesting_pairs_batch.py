# This script scans daily parquet files one at a time with a shared row budget,
# summarizes valid H3 A-to-B trips, and optionally maps selected route pairs.
from __future__ import annotations

import argparse
import csv
import os
import resource
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import branca.colormap as cm
import folium
import h3.api.basic_int as h3
import numpy as np
import pandas as pd
import pyarrow.dataset as ds


DATA_DIR = Path("data_files")
EXCLUDED_PARQUET = Path("data_files/2019_01_01.parquet")
TOTAL_ROWS = 50_000_000
ROWS_PER_FILE = 0
MAX_FILES = 0
LAT_COL = "latitude"
LON_COL = "longitude"
USER_COL = "caid"
TIME_COL = "utc_timestamp"

RES = 9
MAX_TRIP_LENGTH = 100
TIME_THRESHOLD = 7200
MIN_KM_DIST = 3.0
MIN_UNIQUE_HEXES_PER_USER = 3
MAX_UNIQUE_HEXES_PER_USER = 10

MIN_TRIPS_FOR_MAP = 5
MIN_PATH_HEXES = 5
MIN_INTERMEDIATE_HEXES_FOR_MAP = 10
MAX_MAPS = 3
TOP_PATHS_PER_MAP = 5

SUMMARY_CSV = Path("interesting_trip_summary.csv")
MAP_OUTPUT_DIR = Path("interesting_pair_maps")
PROFILE_CSV = Path("interesting_pairs_profile.csv")
ZOOM_START = 13

MONTH_ORDER = {
    "january": 1,
    "february": 2,
    "march": 3,
}
ROUTE_COLORS = ["#0057e7", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


@dataclass
class PairAggregate:
    trip_count: int = 0
    total_intermediate_len: int = 0
    max_intermediate_len: int = 0
    intermediate_cells: set[int] = field(default_factory=set)


@dataclass
class PairDetail:
    trip_count: int = 0
    cell_visit_counts: Counter[int] = field(default_factory=Counter)
    path_counts: Counter[tuple[int, ...]] = field(default_factory=Counter)


@dataclass
class UserSequence:
    cells: np.ndarray
    timestamps: np.ndarray


@dataclass
class HexLookupTable:
    cell_to_code: dict[int, int] = field(default_factory=dict)
    code_to_cell: list[int] = field(default_factory=list)
    latlng_cache: dict[int, tuple[float, float]] = field(default_factory=dict)
    distance_cache: dict[tuple[int, int], float] = field(default_factory=dict)

    def encode_cell(self, cell: int) -> int:
        cell_int = int(cell)
        existing = self.cell_to_code.get(cell_int)
        if existing is not None:
            return existing

        code = len(self.code_to_cell)
        if code >= np.iinfo(np.uint32).max:
            raise OverflowError("Hex lookup table exceeded uint32 capacity.")

        self.cell_to_code[cell_int] = code
        self.code_to_cell.append(cell_int)
        return code

    def decode_cell(self, code: int) -> int:
        return self.code_to_cell[int(code)]

    def cell_to_latlng(self, code: int) -> tuple[float, float]:
        code_int = int(code)
        cached = self.latlng_cache.get(code_int)
        if cached is not None:
            return cached

        latlng = h3.cell_to_latlng(self.decode_cell(code_int))
        self.latlng_cache[code_int] = latlng
        return latlng

    def haversine_km(self, code_a: int, code_b: int) -> float:
        key = (int(code_a), int(code_b))
        cached = self.distance_cache.get(key)
        if cached is not None:
            return cached

        lat1, lon1 = self.cell_to_latlng(key[0])
        lat2, lon2 = self.cell_to_latlng(key[1])
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        term = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        distance = 2 * 6371 * atan2(sqrt(term), sqrt(1 - term))
        self.distance_cache[key] = distance
        self.distance_cache[(key[1], key[0])] = distance
        return distance


@dataclass
class PreparedFrameStats:
    rows_loaded: int
    valid_rows: int
    total_caids: int
    frame_memory_mb: float


@dataclass
class SequenceBuildStats:
    qualifying_caids: int = 0
    qualifying_rows: int = 0
    deduped_rows: int = 0
    sequence_count: int = 0
    total_sequence_points: int = 0
    max_sequence_len: int = 0


PROFILE_FIELDNAMES = [
    "day",
    "source_file",
    "rows_loaded",
    "valid_rows",
    "total_caids",
    "frame_memory_mb",
    "target_unique_hex_count",
    "qualifying_caids",
    "qualifying_rows",
    "deduped_rows",
    "sequence_count",
    "total_sequence_points",
    "max_sequence_len",
    "pair_count",
    "trip_instances",
    "build_seconds",
    "pair_aggregation_seconds",
    "total_seconds",
    "current_rss_mb",
    "peak_rss_mb",
]


HEX_LOOKUP = HexLookupTable()


def reset_runtime_state() -> None:
    global HEX_LOOKUP
    HEX_LOOKUP = HexLookupTable()


def get_hex_lookup() -> HexLookupTable:
    return HEX_LOOKUP


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
            "Summarize A-to-B H3 trips across daily parquet files and map "
            "the strongest candidate route pairs."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--parquet-file",
        type=Path,
        default=None,
        help="Process one specific parquet file instead of scanning all month folders.",
    )
    parser.add_argument(
        "--total-rows",
        type=int,
        default=TOTAL_ROWS,
        help="Total row budget spread across all selected parquet files. Use 0 for no total cap.",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=ROWS_PER_FILE,
        help=(
            "Optional per-file row cap. Use 0 to automatically split --total-rows "
            "across the selected files."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=MAX_FILES,
        help="Maximum number of parquet files to process. Use 0 for all files.",
    )
    parser.add_argument(
        "--min-unique-hexes-per-user",
        type=int,
        default=MIN_UNIQUE_HEXES_PER_USER,
        help="Minimum unique H3 hexagons per CAID, inclusive.",
    )
    parser.add_argument(
        "--max-unique-hexes-per-user",
        type=int,
        default=MAX_UNIQUE_HEXES_PER_USER,
        help="Maximum unique H3 hexagons per CAID, inclusive.",
    )
    parser.add_argument(
        "--time-threshold",
        type=float,
        default=TIME_THRESHOLD,
        help="Maximum seconds allowed between trip start and end.",
    )
    parser.add_argument(
        "--max-trip-length",
        type=int,
        default=MAX_TRIP_LENGTH,
        help="Maximum number of distinct consecutive H3 cells in a candidate trip.",
    )
    parser.add_argument(
        "--min-path-hexes",
        type=int,
        default=MIN_PATH_HEXES,
        help="Minimum total H3 hexagons required in a valid A-to-B trip path.",
    )
    parser.add_argument(
        "--min-km-dist",
        type=float,
        default=MIN_KM_DIST,
        help="Minimum distance between start and end hex centers.",
    )
    parser.add_argument("--min-trips", type=int, default=MIN_TRIPS_FOR_MAP)
    parser.add_argument(
        "--min-intermediate-hexes",
        type=int,
        default=MIN_INTERMEDIATE_HEXES_FOR_MAP,
    )
    parser.add_argument(
        "--max-maps",
        type=int,
        default=MAX_MAPS,
        help="Number of selected A-to-B pairs to map. Use 0 for summary-only mode.",
    )
    parser.add_argument("--top-paths", type=int, default=TOP_PATHS_PER_MAP)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--map-dir", type=Path, default=MAP_OUTPUT_DIR)
    parser.add_argument(
        "--profile-unique-hex-counts",
        type=parse_int_list,
        default=None,
        help=(
            "Comma-separated exact unique-hex counts to benchmark per parquet file. "
            "When set, the script writes a profiling CSV and skips map generation."
        ),
    )
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=PROFILE_CSV,
        help="CSV output path for --profile-unique-hex-counts mode.",
    )
    return parser.parse_args()


def parquet_sort_key(path: Path, data_dir: Path) -> tuple[int, int, str]:
    relative = path.relative_to(data_dir)
    if len(relative.parts) >= 2:
        month_name = relative.parts[0].lower()
        day_token = Path(relative.parts[-1]).stem
        try:
            day_num = int(day_token)
        except ValueError:
            day_num = 999
        return (MONTH_ORDER.get(month_name, 999), day_num, relative.as_posix())
    return (999, 999, relative.as_posix())


def relative_day_label(path: Path, data_dir: Path) -> str:
    relative = path.relative_to(data_dir)
    if len(relative.parts) >= 2:
        return f"{relative.parts[0]}/{relative.stem}"
    return relative.stem


def list_parquet_files(
    data_dir: Path,
    parquet_file: Path | None,
    max_files: int,
) -> list[Path]:
    if parquet_file is not None:
        return [parquet_file]

    excluded = EXCLUDED_PARQUET.resolve()
    files = [
        path
        for path in data_dir.rglob("*.parquet")
        if path.resolve() != excluded
    ]
    files = sorted(files, key=lambda path: parquet_sort_key(path, data_dir))
    if max_files > 0:
        files = files[:max_files]
    return files


def read_parquet_subset(path: Path, nrows: int | None = None, columns=None) -> pd.DataFrame:
    try:
        dataset = ds.dataset(str(path), format="parquet")
        if nrows is None:
            table = dataset.to_table(columns=columns)
        else:
            table = dataset.scanner(columns=columns).head(nrows)
        return table.to_pandas()
    except Exception:
        frame = pd.read_parquet(path, columns=columns)
        return frame.head(nrows) if nrows is not None else frame


def current_rss_mb() -> float:
    try:
        rss_kb = int(
            subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                text=True,
            ).strip()
        )
    except Exception:
        return peak_rss_mb()
    return rss_kb / 1024.0


def peak_rss_mb() -> float:
    raw_value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if raw_value <= 0:
        return 0.0
    # macOS reports bytes; Linux typically reports kilobytes.
    if raw_value > 10_000_000:
        return raw_value / (1024.0 * 1024.0)
    return raw_value / 1024.0


def row_cap_for_file(
    file_index: int,
    total_files: int,
    total_rows: int,
    rows_per_file: int,
) -> int | None:
    if rows_per_file > 0:
        return rows_per_file
    if total_rows <= 0:
        return None

    base_rows, extra_rows = divmod(total_rows, total_files)
    return base_rows + (1 if file_index <= extra_rows else 0)


def prepare_sequence_frame(
    frame: pd.DataFrame,
    res: int,
) -> tuple[pd.DataFrame, PreparedFrameStats]:
    rows_loaded = len(frame)
    frame = frame[[USER_COL, TIME_COL, LAT_COL, LON_COL]].copy()
    frame[LAT_COL] = pd.to_numeric(frame[LAT_COL], errors="coerce")
    frame[LON_COL] = pd.to_numeric(frame[LON_COL], errors="coerce")
    frame[TIME_COL] = pd.to_numeric(frame[TIME_COL], errors="coerce")
    frame = frame.dropna(subset=[USER_COL, TIME_COL, LAT_COL, LON_COL])
    valid_rows = len(frame)
    total_caids = int(frame[USER_COL].nunique()) if not frame.empty else 0
    if frame.empty:
        return (
            frame.assign(
                cell=pd.Series(dtype=np.uint32),
                unique_hex_count=pd.Series(dtype=np.int64),
            ),
            PreparedFrameStats(
                rows_loaded=rows_loaded,
                valid_rows=valid_rows,
                total_caids=total_caids,
                frame_memory_mb=0.0,
            ),
        )

    frame = frame.sort_values([USER_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    hex_lookup = get_hex_lookup()
    frame["cell"] = np.fromiter(
        (
            hex_lookup.encode_cell(h3.latlng_to_cell(float(lat), float(lon), res))
            for lat, lon in frame[[LAT_COL, LON_COL]].itertuples(index=False, name=None)
        ),
        dtype=np.uint32,
        count=len(frame),
    )
    frame["unique_hex_count"] = frame.groupby(USER_COL, sort=False)["cell"].transform("nunique")
    stats = PreparedFrameStats(
        rows_loaded=rows_loaded,
        valid_rows=valid_rows,
        total_caids=total_caids,
        frame_memory_mb=float(frame.memory_usage(index=True, deep=True).sum() / (1024**2)),
    )
    return frame, stats


def dedupe_prepared_sequence_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.loc[:, [USER_COL, TIME_COL, "cell", "unique_hex_count"]].copy()

    previous_user = frame[USER_COL].shift()
    previous_cell = frame["cell"].shift()
    return frame.loc[
        frame[USER_COL].ne(previous_user) | frame["cell"].ne(previous_cell),
        [USER_COL, TIME_COL, "cell", "unique_hex_count"],
    ].copy()


def build_sequences_from_deduped_frame(
    frame: pd.DataFrame,
) -> tuple[dict[str, UserSequence], int]:
    if frame.empty:
        return {}, 0

    sequences: dict[str, UserSequence] = {}
    max_sequence_len = 0
    grouped = frame.groupby(USER_COL, sort=False, observed=True)
    for user, group in grouped:
        cells = group["cell"].to_numpy(dtype=np.uint32)
        timestamps = group[TIME_COL].to_numpy(dtype=np.float64)
        sequences[str(user)] = UserSequence(cells=cells, timestamps=timestamps)
        if len(cells) > max_sequence_len:
            max_sequence_len = len(cells)
    return sequences, max_sequence_len


def build_user_sequences_from_prepared_frame(
    frame: pd.DataFrame,
    min_unique_hexes_per_user: int,
    max_unique_hexes_per_user: int,
    *,
    exact_unique_hex_count: int | None = None,
) -> tuple[dict[str, UserSequence], SequenceBuildStats]:
    if exact_unique_hex_count is not None:
        filtered = frame.loc[
            frame["unique_hex_count"] == exact_unique_hex_count,
            [USER_COL, TIME_COL, "cell", "unique_hex_count"],
        ].copy()
    else:
        filtered = frame.loc[
            (frame["unique_hex_count"] >= min_unique_hexes_per_user)
            & (frame["unique_hex_count"] <= max_unique_hexes_per_user),
            [USER_COL, TIME_COL, "cell", "unique_hex_count"],
        ].copy()

    stats = SequenceBuildStats(
        qualifying_caids=int(filtered[USER_COL].nunique()) if not filtered.empty else 0,
        qualifying_rows=len(filtered),
    )
    if filtered.empty:
        return {}, stats

    filtered = dedupe_prepared_sequence_frame(filtered)
    stats.deduped_rows = len(filtered)
    sequences, stats.max_sequence_len = build_sequences_from_deduped_frame(
        filtered.loc[:, [USER_COL, TIME_COL, "cell"]]
    )
    stats.sequence_count = len(sequences)
    stats.total_sequence_points = stats.deduped_rows
    return sequences, stats


def build_user_sequences_with_time(
    frame: pd.DataFrame,
    res: int,
    min_unique_hexes_per_user: int,
    max_unique_hexes_per_user: int,
) -> dict[str, UserSequence]:
    prepared_frame, _ = prepare_sequence_frame(frame, res)
    sequences, _ = build_user_sequences_from_prepared_frame(
        prepared_frame,
        min_unique_hexes_per_user,
        max_unique_hexes_per_user,
    )
    return sequences


def haversine_km(cell_a: int, cell_b: int) -> float:
    return get_hex_lookup().haversine_km(cell_a, cell_b)


def iter_valid_trip_ranges(
    sequence: UserSequence,
    time_threshold: float,
    max_trip_length: int,
    min_path_hexes: int,
    min_km_dist: float,
):
    cells = sequence.cells
    timestamps = sequence.timestamps

    for start_idx in range(len(cells)):
        start_cell = int(cells[start_idx])
        start_time = timestamps[start_idx]
        seen = {start_cell}

        stop_idx = min(len(cells), start_idx + max_trip_length)
        for end_idx in range(start_idx + 1, stop_idx):
            end_cell = int(cells[end_idx])
            if timestamps[end_idx] - start_time > time_threshold:
                break
            if end_cell in seen:
                break

            seen.add(end_cell)
            path_length = end_idx - start_idx + 1
            if path_length < min_path_hexes:
                continue
            if haversine_km(start_cell, end_cell) < min_km_dist:
                continue

            yield cells, start_idx, end_idx


def update_pair_aggregates(
    sequences: dict[str, UserSequence],
    pair_aggregates: dict[tuple[int, int], PairAggregate],
    time_threshold: float,
    max_trip_length: int,
    min_path_hexes: int,
    min_km_dist: float,
) -> None:
    for sequence in sequences.values():
        for cells, start_idx, end_idx in iter_valid_trip_ranges(
            sequence, time_threshold, max_trip_length, min_path_hexes, min_km_dist
        ):
            pair = (int(cells[start_idx]), int(cells[end_idx]))
            intermediates = cells[start_idx + 1 : end_idx]
            stats = pair_aggregates[pair]
            stats.trip_count += 1
            stats.total_intermediate_len += len(intermediates)
            stats.max_intermediate_len = max(stats.max_intermediate_len, len(intermediates))
            stats.intermediate_cells.update(map(int, intermediates))


def update_pair_details(
    sequences: dict[str, UserSequence],
    selected_pairs: set[tuple[int, int]],
    pair_details: dict[tuple[int, int], PairDetail],
    time_threshold: float,
    max_trip_length: int,
    min_path_hexes: int,
    min_km_dist: float,
) -> None:
    if not selected_pairs:
        return

    for sequence in sequences.values():
        for cells, start_idx, end_idx in iter_valid_trip_ranges(
            sequence, time_threshold, max_trip_length, min_path_hexes, min_km_dist
        ):
            pair = (int(cells[start_idx]), int(cells[end_idx]))
            if pair not in selected_pairs:
                continue

            path = tuple(map(int, cells[start_idx : end_idx + 1]))
            stats = pair_details[pair]
            stats.trip_count += 1
            stats.cell_visit_counts.update(path)
            stats.path_counts[path] += 1


def unique_intermediate_count(stats: PairAggregate) -> int:
    return len(stats.intermediate_cells)


def build_trip_summary_frame(
    pair_aggregates: dict[tuple[int, int], PairAggregate],
) -> pd.DataFrame:
    hex_lookup = get_hex_lookup()
    records = [
        {
            "start_hex_code": hex_a,
            "end_hex_code": hex_b,
            "start_hex_int": hex_lookup.decode_cell(hex_a),
            "end_hex_int": hex_lookup.decode_cell(hex_b),
            "trip_count": stats.trip_count,
            "unique_intermediate_hexes": unique_intermediate_count(stats),
            "avg_intermediate_len": (
                stats.total_intermediate_len / stats.trip_count if stats.trip_count else 0.0
            ),
            "max_intermediate_len": stats.max_intermediate_len,
        }
        for (hex_a, hex_b), stats in pair_aggregates.items()
    ]
    if not records:
        return pd.DataFrame(
            columns=[
                "start_hex_code",
                "end_hex_code",
                "start_hex_int",
                "end_hex_int",
                "trip_count",
                "unique_intermediate_hexes",
                "avg_intermediate_len",
                "max_intermediate_len",
                "start_hex",
                "end_hex",
            ]
        )

    summary = pd.DataFrame.from_records(records).sort_values(
        ["trip_count", "unique_intermediate_hexes", "max_intermediate_len"],
        ascending=False,
        kind="mergesort",
    )
    summary["start_hex"] = summary["start_hex_int"].apply(h3.int_to_str)
    summary["end_hex"] = summary["end_hex_int"].apply(h3.int_to_str)
    return summary.reset_index(drop=True)


def write_trip_summary_csv(summary: pd.DataFrame, out_path: Path) -> None:
    export_columns = [
        "start_hex",
        "end_hex",
        "trip_count",
        "unique_intermediate_hexes",
        "avg_intermediate_len",
        "max_intermediate_len",
    ]
    summary.loc[:, export_columns].to_csv(out_path, index=False, float_format="%.2f")
    print(f"Saved trip summary -> {out_path.resolve()} ({len(summary):,} rows)")


def select_interesting_pairs(
    summary: pd.DataFrame,
    min_trips: int,
    min_intermediate_hexes: int,
    max_maps: int,
) -> pd.DataFrame:
    if summary.empty or max_maps <= 0:
        return summary.head(0)

    selected = summary.loc[
        (summary["trip_count"] >= min_trips)
        & (summary["unique_intermediate_hexes"] >= min_intermediate_hexes)
    ].sort_values(
        ["unique_intermediate_hexes", "trip_count", "max_intermediate_len"],
        ascending=False,
        kind="mergesort",
    )
    return selected.head(max_maps).reset_index(drop=True)


def hex_boundary_geojson(cell: int) -> list[tuple[float, float]]:
    boundary = h3.cell_to_boundary(cell)
    coords = [(lon, lat) for lat, lon in boundary]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def build_map(
    pair: tuple[int, int],
    detail: PairDetail,
    out_path: Path,
    top_paths: int,
) -> None:
    if detail.trip_count == 0:
        return

    hex_lookup = get_hex_lookup()
    hex_a = hex_lookup.decode_cell(pair[0])
    hex_b = hex_lookup.decode_cell(pair[1])
    hex_probs = {
        hex_lookup.decode_cell(cell): count / detail.trip_count
        for cell, count in detail.cell_visit_counts.items()
    }
    ranked_paths = [
        (tuple(hex_lookup.decode_cell(cell) for cell in path), count)
        for path, count in detail.path_counts.most_common(top_paths)
    ]

    lat_a, lon_a = h3.cell_to_latlng(hex_a)
    lat_b, lon_b = h3.cell_to_latlng(hex_b)
    center = ((lat_a + lat_b) / 2, (lon_a + lon_b) / 2)
    fmap = folium.Map(location=list(center), zoom_start=ZOOM_START, tiles="CartoDB Positron")

    all_path_cells = {cell for path, _ in ranked_paths for cell in path}
    path_probs = {cell: hex_probs[cell] for cell in all_path_cells if cell in hex_probs}

    colormap = None
    if path_probs:
        min_prob, max_prob = min(path_probs.values()), max(path_probs.values())
        if min_prob == max_prob:
            max_prob = min_prob + 1e-9
        colormap = cm.linear.YlOrRd_09.scale(min_prob, max_prob)
        colormap.caption = "P(H) - visit probability for selected A->B pair"
        colormap.add_to(fmap)

    for cell in all_path_cells:
        latlon = [(lat, lon) for lon, lat in hex_boundary_geojson(cell)]
        probability = path_probs.get(cell, 0.0)
        folium.Polygon(
            locations=latlon,
            color="#333333",
            weight=0.8,
            fill=True,
            fill_color=colormap(probability) if colormap else "#3388ff",
            fill_opacity=0.65,
            tooltip=f"{h3.int_to_str(cell)} | P(H)={probability:.4f}",
        ).add_to(fmap)

    for rank, (path, count) in enumerate(ranked_paths, start=1):
        folium.PolyLine(
            locations=[h3.cell_to_latlng(cell) for cell in path],
            color=ROUTE_COLORS[(rank - 1) % len(ROUTE_COLORS)],
            weight=5,
            opacity=0.8,
            tooltip=f"Route #{rank}: {count}/{detail.trip_count} trips ({count / detail.trip_count:.1%})",
        ).add_to(fmap)

    folium.Marker(
        h3.cell_to_latlng(hex_a),
        tooltip=f"A: {h3.int_to_str(hex_a)}",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(fmap)
    folium.Marker(
        h3.cell_to_latlng(hex_b),
        tooltip=f"B: {h3.int_to_str(hex_b)}",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(fmap)

    fmap.save(str(out_path))


def summarize_detail(pair: tuple[int, int], detail: PairDetail, top_paths: int) -> None:
    hex_lookup = get_hex_lookup()
    start_hex = h3.int_to_str(hex_lookup.decode_cell(pair[0]))
    end_hex = h3.int_to_str(hex_lookup.decode_cell(pair[1]))
    print(
        f"\nPair {start_hex} -> {end_hex}: "
        f"{detail.trip_count} trips, {len(detail.path_counts)} distinct routes"
    )
    for rank, (path, count) in enumerate(detail.path_counts.most_common(top_paths), start=1):
        path_text = " -> ".join(h3.int_to_str(hex_lookup.decode_cell(cell)) for cell in path)
        print(f"  Route #{rank}: {count}/{detail.trip_count} trips ({count / detail.trip_count:.1%})")
        print(f"    {path_text}")


def process_file(
    path: Path,
    data_dir: Path,
    rows_per_file: int | None,
    min_unique_hexes_per_user: int,
    max_unique_hexes_per_user: int,
) -> dict[str, UserSequence]:
    columns = [USER_COL, TIME_COL, LAT_COL, LON_COL]
    frame = read_parquet_subset(path, nrows=rows_per_file, columns=columns)
    day_label = relative_day_label(path, data_dir)
    print(f"Loaded {day_label}: {len(frame):,} rows")
    prepared_frame, prepared_stats = prepare_sequence_frame(frame, RES)
    sequences, sequence_stats = build_user_sequences_from_prepared_frame(
        prepared_frame,
        min_unique_hexes_per_user,
        max_unique_hexes_per_user,
    )
    print(
        f"  Built {len(sequences):,} user sequences after filtering caids to "
        f"{min_unique_hexes_per_user}-{max_unique_hexes_per_user} unique hexes"
    )
    print(
        f"  Valid rows={prepared_stats.valid_rows:,}, total_caids={prepared_stats.total_caids:,}, "
        f"qualifying_caids={sequence_stats.qualifying_caids:,}, deduped_rows={sequence_stats.deduped_rows:,}"
    )
    return sequences


def initialize_profile_csv(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDNAMES)
        writer.writeheader()


def append_profile_record(record: dict[str, int | float | str], out_path: Path) -> None:
    with out_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDNAMES)
        writer.writerow(record)


def profile_exact_unique_hex_counts(
    parquet_files: list[Path],
    args: argparse.Namespace,
) -> None:
    started = time.time()
    record_count = 0
    initialize_profile_csv(args.profile_csv)

    print(
        "Profiling exact unique-hex counts: "
        + ", ".join(str(value) for value in args.profile_unique_hex_counts)
    )
    for index, path in enumerate(parquet_files, start=1):
        rows_for_file = row_cap_for_file(
            index,
            len(parquet_files),
            args.total_rows,
            args.rows_per_file,
        )
        day_label = relative_day_label(path, args.data_dir)
        print(f"\nProfile file {index}/{len(parquet_files)}: {day_label}")
        frame = read_parquet_subset(
            path,
            nrows=rows_for_file,
            columns=[USER_COL, TIME_COL, LAT_COL, LON_COL],
        )
        prepared_frame, prepared_stats = prepare_sequence_frame(frame, RES)
        deduped_prepared_frame = dedupe_prepared_sequence_frame(prepared_frame)
        target_counts = set(args.profile_unique_hex_counts)
        qualifying_rows_by_count = (
            prepared_frame.loc[prepared_frame["unique_hex_count"].isin(target_counts)]
            .groupby("unique_hex_count", observed=True)
            .size()
            .to_dict()
        )
        qualifying_caids_by_count = (
            prepared_frame.loc[prepared_frame["unique_hex_count"].isin(target_counts)]
            .groupby("unique_hex_count", observed=True)[USER_COL]
            .nunique()
            .to_dict()
        )
        deduped_rows_by_count = (
            deduped_prepared_frame.loc[deduped_prepared_frame["unique_hex_count"].isin(target_counts)]
            .groupby("unique_hex_count", observed=True)
            .size()
            .to_dict()
        )
        print(
            f"  Loaded rows={prepared_stats.rows_loaded:,}, valid_rows={prepared_stats.valid_rows:,}, "
            f"total_caids={prepared_stats.total_caids:,}, frame_memory={prepared_stats.frame_memory_mb:.1f} MB"
        )

        for unique_hex_count in args.profile_unique_hex_counts:
            build_started = time.perf_counter()
            deduped_subset = deduped_prepared_frame.loc[
                deduped_prepared_frame["unique_hex_count"] == unique_hex_count,
                [USER_COL, TIME_COL, "cell"],
            ]
            sequences, max_sequence_len = build_sequences_from_deduped_frame(
                deduped_subset
            )
            build_seconds = time.perf_counter() - build_started

            pair_started = time.perf_counter()
            pair_aggregates: dict[tuple[int, int], PairAggregate] = defaultdict(PairAggregate)
            update_pair_aggregates(
                sequences,
                pair_aggregates,
                args.time_threshold,
                args.max_trip_length,
                args.min_path_hexes,
                args.min_km_dist,
            )
            pair_seconds = time.perf_counter() - pair_started

            total_trip_instances = sum(stats.trip_count for stats in pair_aggregates.values())
            total_seconds = build_seconds + pair_seconds
            sequence_count = len(sequences)
            deduped_rows = int(deduped_rows_by_count.get(unique_hex_count, 0))
            record = {
                "day": day_label,
                "source_file": str(path),
                "rows_loaded": prepared_stats.rows_loaded,
                "valid_rows": prepared_stats.valid_rows,
                "total_caids": prepared_stats.total_caids,
                "frame_memory_mb": round(prepared_stats.frame_memory_mb, 3),
                "target_unique_hex_count": unique_hex_count,
                "qualifying_caids": int(qualifying_caids_by_count.get(unique_hex_count, 0)),
                "qualifying_rows": int(qualifying_rows_by_count.get(unique_hex_count, 0)),
                "deduped_rows": deduped_rows,
                "sequence_count": sequence_count,
                "total_sequence_points": deduped_rows,
                "max_sequence_len": max_sequence_len,
                "pair_count": len(pair_aggregates),
                "trip_instances": total_trip_instances,
                "build_seconds": round(build_seconds, 6),
                "pair_aggregation_seconds": round(pair_seconds, 6),
                "total_seconds": round(total_seconds, 6),
                "current_rss_mb": round(current_rss_mb(), 3),
                "peak_rss_mb": round(peak_rss_mb(), 3),
            }
            append_profile_record(record, args.profile_csv)
            record_count += 1
            print(
                f"  hexes={unique_hex_count:>2} -> caids={record['qualifying_caids']:,}, "
                f"sequences={sequence_count:,}, pairs={len(pair_aggregates):,}, "
                f"time={total_seconds:.2f}s, peak_rss={record['peak_rss_mb']:.1f} MB"
            )

    print(f"\nSaved profiling CSV -> {args.profile_csv.resolve()} ({record_count:,} rows)")
    print(f"Total time: {time.time() - started:.2f} s")


def main() -> None:
    args = parse_args()
    started = time.time()
    reset_runtime_state()

    if args.min_unique_hexes_per_user > args.max_unique_hexes_per_user:
        raise ValueError("--min-unique-hexes-per-user cannot exceed --max-unique-hexes-per-user")
    if args.min_path_hexes > args.max_trip_length:
        raise ValueError("--min-path-hexes cannot exceed --max-trip-length")
    if args.profile_unique_hex_counts is not None and any(value < 1 for value in args.profile_unique_hex_counts):
        raise ValueError("--profile-unique-hex-counts values must all be positive integers")

    parquet_files = list_parquet_files(args.data_dir, args.parquet_file, args.max_files)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {args.data_dir.resolve()}")

    if args.parquet_file is not None:
        print(f"Scanning 1 parquet file: {args.parquet_file.resolve()}")
    else:
        print(
            f"Scanning {len(parquet_files)} parquet file(s) from {args.data_dir.resolve()} "
            f"(excluding {EXCLUDED_PARQUET.name})"
        )

    if args.profile_unique_hex_counts is not None:
        profile_exact_unique_hex_counts(parquet_files, args)
        return

    pair_aggregates: dict[tuple[int, int], PairAggregate] = defaultdict(PairAggregate)
    total_sequences = 0
    for index, path in enumerate(parquet_files, start=1):
        print(f"\nSummary pass file {index}/{len(parquet_files)}")
        rows_for_file = row_cap_for_file(
            index,
            len(parquet_files),
            args.total_rows,
            args.rows_per_file,
        )
        sequences = process_file(
            path,
            args.data_dir,
            rows_for_file,
            args.min_unique_hexes_per_user,
            args.max_unique_hexes_per_user,
        )
        total_sequences += len(sequences)
        update_pair_aggregates(
            sequences,
            pair_aggregates,
            args.time_threshold,
            args.max_trip_length,
            args.min_path_hexes,
            args.min_km_dist,
        )
        print(f"  Running A->B pair count: {len(pair_aggregates):,}")

    print(f"\nSummary pass complete: {len(pair_aggregates):,} A->B pairs from {total_sequences:,} sequences")
    summary = build_trip_summary_frame(pair_aggregates)
    write_trip_summary_csv(summary, args.summary_csv)

    interesting_pairs = select_interesting_pairs(
        summary,
        min_trips=args.min_trips,
        min_intermediate_hexes=args.min_intermediate_hexes,
        max_maps=args.max_maps,
    )
    if interesting_pairs.empty:
        print(
            "\nNo maps requested or no pairs met the thresholds. "
            "The summary CSV is still ready for downstream calculations."
        )
        print(f"Total time: {time.time() - started:.2f} s")
        return

    print("\nSelected pairs for mapping:")
    selected_keys = set()
    for row in interesting_pairs.itertuples(index=False):
        pair = (int(row.start_hex_code), int(row.end_hex_code))
        selected_keys.add(pair)
        print(
            f"  {row.start_hex} -> {row.end_hex} | "
            f"trips={row.trip_count}, unique_intermediate_hexes={row.unique_intermediate_hexes}"
        )

    pair_details: dict[tuple[int, int], PairDetail] = defaultdict(PairDetail)
    for index, path in enumerate(parquet_files, start=1):
        print(f"\nDetail pass file {index}/{len(parquet_files)}")
        rows_for_file = row_cap_for_file(
            index,
            len(parquet_files),
            args.total_rows,
            args.rows_per_file,
        )
        sequences = process_file(
            path,
            args.data_dir,
            rows_for_file,
            args.min_unique_hexes_per_user,
            args.max_unique_hexes_per_user,
        )
        update_pair_details(
            sequences,
            selected_keys,
            pair_details,
            args.time_threshold,
            args.max_trip_length,
            args.min_path_hexes,
            args.min_km_dist,
        )

    args.map_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(interesting_pairs.itertuples(index=False), start=1):
        pair = (int(row.start_hex_code), int(row.end_hex_code))
        detail = pair_details[pair]
        summarize_detail(pair, detail, args.top_paths)
        out_name = f"{rank:02d}_{row.start_hex}_to_{row.end_hex}.html"
        out_path = args.map_dir / out_name
        build_map(pair, detail, out_path, args.top_paths)
        print(f"  Saved map -> {out_path.resolve()}")

    print(f"\nTotal time: {time.time() - started:.2f} s")


if __name__ == "__main__":
    main()
