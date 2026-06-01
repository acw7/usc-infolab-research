# This script is the earlier control version of the A-to-B H3 trip batch workflow.
# It summarizes candidate route pairs and maps selected pairs from a limited file set.
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import branca.colormap as cm
import folium
import h3.api.basic_int as h3
import numpy as np
import pandas as pd
import pyarrow.dataset as ds


DATA_DIR = Path("data_files")
DEFAULT_PARQUET_FILE = Path("data_files/2019_01_01.parquet")
ROWS_PER_FILE = 10_000_000
MAX_FILES = 1
LAT_COL = "latitude"
LON_COL = "longitude"
USER_COL = "caid"
TIME_COL = "utc_timestamp"

RES = 9
MAX_TRIP_LENGTH = 100
TIME_THRESHOLD = 7200
MIN_KM_DIST = 3.0
# Drop users that only touch a tiny number of hexes so we spend less time
# generating candidate trips for stationary / low-signal caids.
MIN_UNIQUE_HEXES_PER_USER = 2

MIN_TRIPS_FOR_MAP = 5
MIN_INTERMEDIATE_HEXES_FOR_MAP = 5
MAX_MAPS = 3
TOP_PATHS_PER_MAP = 5

SUMMARY_CSV = Path("interesting_trip_summary.csv")
MAP_OUTPUT_DIR = Path("interesting_pair_maps")
ZOOM_START = 13

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize A->B trips across parquet files and map interesting pairs."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--parquet-file",
        type=Path,
        default=DEFAULT_PARQUET_FILE,
        help="Process one specific parquet file instead of scanning a directory.",
    )
    parser.add_argument("--rows-per-file", type=int, default=ROWS_PER_FILE)
    parser.add_argument("--max-files", type=int, default=MAX_FILES)
    parser.add_argument(
        "--min-unique-hexes-per-user",
        type=int,
        default=MIN_UNIQUE_HEXES_PER_USER,
        help="Prune caids with <= this many unique hexagons before trip generation.",
    )
    parser.add_argument("--min-trips", type=int, default=MIN_TRIPS_FOR_MAP)
    parser.add_argument(
        "--min-intermediate-hexes",
        type=int,
        default=MIN_INTERMEDIATE_HEXES_FOR_MAP,
    )
    parser.add_argument("--max-maps", type=int, default=MAX_MAPS)
    parser.add_argument("--top-paths", type=int, default=TOP_PATHS_PER_MAP)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--map-dir", type=Path, default=MAP_OUTPUT_DIR)
    return parser.parse_args()


def list_parquet_files(
    data_dir: Path,
    parquet_file: Path | None,
    max_files: int | None,
) -> list[Path]:
    if parquet_file is not None:
        return [parquet_file]

    files = sorted(data_dir.glob("*.parquet"))
    if max_files is not None:
        files = files[:max_files]
    return files


def read_parquet_subset(path: Path, nrows=None, columns=None) -> pd.DataFrame:
    try:
        dataset = ds.dataset(str(path), format="parquet")
        if nrows is None:
            table = dataset.to_table(columns=columns)
        else:
            table = dataset.scanner(columns=columns).head(nrows)
        return table.to_pandas()
    except Exception:
        frame = pd.read_parquet(path, columns=columns)
        return frame.head(nrows) if nrows else frame


def build_user_sequences_with_time(
    frame: pd.DataFrame,
    res: int,
    min_unique_hexes_per_user: int,
) -> dict[str, UserSequence]:
    frame = frame[[USER_COL, TIME_COL, LAT_COL, LON_COL]].copy()
    frame[LAT_COL] = pd.to_numeric(frame[LAT_COL], errors="coerce")
    frame[LON_COL] = pd.to_numeric(frame[LON_COL], errors="coerce")
    frame[TIME_COL] = pd.to_numeric(frame[TIME_COL], errors="coerce")
    frame = frame.dropna(subset=[USER_COL, TIME_COL, LAT_COL, LON_COL])
    if frame.empty:
        return {}

    frame = frame.sort_values([USER_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    # Build the H3 cell column once up front so later filtering/grouping stays in pandas.
    frame["cell"] = np.fromiter(
        (
            h3.latlng_to_cell(float(lat), float(lon), res)
            for lat, lon in frame[[LAT_COL, LON_COL]].itertuples(index=False, name=None)
        ),
        dtype=np.uint64,
        count=len(frame),
    )
    # Prune caids with too few unique hexes before we do any trip enumeration.
    frame["unique_hex_count"] = frame.groupby(USER_COL, sort=False)["cell"].transform("nunique")
    frame = frame.loc[
        frame["unique_hex_count"] > min_unique_hexes_per_user,
        [USER_COL, TIME_COL, "cell"],
    ].copy()
    if frame.empty:
        return {}

    # Collapse consecutive duplicate cells with a vectorized mask instead of a Python row loop.
    previous_user = frame[USER_COL].shift()
    previous_cell = frame["cell"].shift()
    frame = frame.loc[
        frame[USER_COL].ne(previous_user) | frame["cell"].ne(previous_cell),
        [USER_COL, TIME_COL, "cell"],
    ]

    grouped = frame.groupby(USER_COL, sort=False, observed=True)
    return {
        str(user): UserSequence(
            cells=group["cell"].to_numpy(dtype=np.uint64),
            timestamps=group[TIME_COL].to_numpy(dtype=np.float64),
        )
        for user, group in grouped
    }


@lru_cache(maxsize=None)
def haversine_km(cell_a: int, cell_b: int) -> float:
    lat1, lon1 = h3.cell_to_latlng(cell_a)
    lat2, lon2 = h3.cell_to_latlng(cell_b)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    term = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * 6371 * atan2(sqrt(term), sqrt(1 - term))


def iter_valid_trip_ranges(
    sequence: UserSequence,
    time_threshold: float,
    max_length: int,
):
    cells = sequence.cells
    timestamps = sequence.timestamps

    for start_idx in range(len(cells)):
        start_cell = int(cells[start_idx])
        start_time = timestamps[start_idx]
        seen = {start_cell}

        stop_idx = min(len(cells), start_idx + max_length)
        for end_idx in range(start_idx + 1, stop_idx):
            end_cell = int(cells[end_idx])
            if timestamps[end_idx] - start_time > time_threshold:
                break
            if end_cell in seen:
                break

            seen.add(end_cell)
            if haversine_km(start_cell, end_cell) < MIN_KM_DIST:
                continue

            yield cells, start_idx, end_idx


def update_pair_aggregates(
    sequences: dict[str, UserSequence],
    pair_aggregates: dict[tuple[int, int], PairAggregate],
) -> None:
    for sequence in sequences.values():
        for cells, start_idx, end_idx in iter_valid_trip_ranges(
            sequence, TIME_THRESHOLD, MAX_TRIP_LENGTH
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
) -> None:
    if not selected_pairs:
        return

    for sequence in sequences.values():
        for cells, start_idx, end_idx in iter_valid_trip_ranges(
            sequence, TIME_THRESHOLD, MAX_TRIP_LENGTH
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
    # Materialize aggregate stats into a dataframe so sorting/filtering/export are vectorized.
    records = [
        {
            "start_hex_int": hex_a,
            "end_hex_int": hex_b,
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
    # Keep the expensive per-row H3 string formatting isolated to the final export columns.
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
    if summary.empty:
        return summary

    # Rank candidate pairs entirely in pandas rather than rebuilding sorted Python lists.
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

    hex_a, hex_b = pair
    hex_probs = {
        cell: count / detail.trip_count for cell, count in detail.cell_visit_counts.items()
    }
    ranked_paths = detail.path_counts.most_common(top_paths)

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
    print(
        f"\nPair {h3.int_to_str(pair[0])} -> {h3.int_to_str(pair[1])}: "
        f"{detail.trip_count} trips, {len(detail.path_counts)} distinct routes"
    )
    for rank, (path, count) in enumerate(detail.path_counts.most_common(top_paths), start=1):
        path_text = " -> ".join(h3.int_to_str(cell) for cell in path)
        print(f"  Route #{rank}: {count}/{detail.trip_count} trips ({count / detail.trip_count:.1%})")
        print(f"    {path_text}")


def process_file(
    path: Path,
    rows_per_file: int | None,
    min_unique_hexes_per_user: int,
) -> dict[str, UserSequence]:
    columns = [USER_COL, TIME_COL, LAT_COL, LON_COL]
    frame = read_parquet_subset(path, nrows=rows_per_file, columns=columns)
    print(f"Loaded {path.name}: {len(frame):,} rows")
    sequences = build_user_sequences_with_time(frame, RES, min_unique_hexes_per_user)
    print(
        f"  Built {len(sequences):,} user sequences "
        f"after pruning caids with <= {min_unique_hexes_per_user} unique hexes"
    )
    return sequences


def main() -> None:
    args = parse_args()
    started = time.time()

    parquet_files = list_parquet_files(args.data_dir, args.parquet_file, args.max_files)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {args.data_dir.resolve()}")

    if args.parquet_file is not None:
        print(f"Scanning 1 parquet file: {args.parquet_file.resolve()}")
    else:
        print(f"Scanning {len(parquet_files)} parquet file(s) from {args.data_dir.resolve()}")

    pair_aggregates: dict[tuple[int, int], PairAggregate] = defaultdict(PairAggregate)
    total_sequences = 0
    for path in parquet_files:
        sequences = process_file(
            path,
            args.rows_per_file,
            args.min_unique_hexes_per_user,
        )
        total_sequences += len(sequences)
        update_pair_aggregates(sequences, pair_aggregates)

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
            "\nNo pairs met the thresholds. "
            "Try lowering --min-trips or --min-intermediate-hexes for an exploratory run."
        )
        print(f"Total time: {time.time() - started:.2f} s")
        return

    print("\nSelected pairs for mapping:")
    selected_keys = set()
    for row in interesting_pairs.itertuples(index=False):
        pair = (int(row.start_hex_int), int(row.end_hex_int))
        selected_keys.add(pair)
        print(
            f"  {row.start_hex} -> {row.end_hex} | "
            f"trips={row.trip_count}, unique_intermediate_hexes={row.unique_intermediate_hexes}"
        )

    pair_details: dict[tuple[int, int], PairDetail] = defaultdict(PairDetail)
    for path in parquet_files:
        sequences = process_file(
            path,
            args.rows_per_file,
            args.min_unique_hexes_per_user,
        )
        update_pair_details(sequences, selected_keys, pair_details)

    args.map_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(interesting_pairs.itertuples(index=False), start=1):
        pair = (int(row.start_hex_int), int(row.end_hex_int))
        detail = pair_details[pair]
        summarize_detail(pair, detail, args.top_paths)
        out_name = f"{rank:02d}_{h3.int_to_str(pair[0])}_to_{h3.int_to_str(pair[1])}.html"
        out_path = args.map_dir / out_name
        build_map(pair, detail, out_path, args.top_paths)
        print(f"  Saved map -> {out_path.resolve()}")

    print(f"\nTotal time: {time.time() - started:.2f} s")


if __name__ == "__main__":
    main()
