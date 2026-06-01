# This script is the earlier control version for broad A-to-B path probability exploration.
# It summarizes candidate trips and maps selected H3 route probabilities from one parquet file.
from pathlib import Path
import time
import math
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import folium
import branca.colormap as cm
import h3
import pyarrow.dataset as ds
from math import radians, sin, cos, sqrt, atan2
import csv

# Data Config
FILE_PATH = Path("data_files/2019_01_01.parquet")
NUM_ROWS = 100_000
LAT_COL = "latitude"
LON_COL = "longitude"
USER_COL = "caid"
TIME_COL = "utc_timestamp"

RES = 9
MAX_TRIP_LENGTH = 100
TIME_THRESHOLD = 7200  # 2 hours cap for valid A->B trip
MIN_KM_DIST = 3.0
MAX_TRIPS_PER_PAIR = 500
# HEX_A_STR = "89283082aa7ffff"
# HEX_B_STR = "89283092e63ffff"
OUT_HTML = Path("path_probability_map.html")
ZOOM_START = 13

# LAT_MIN, LAT_MAX = 36.80, 38.40
# LON_MIN, LON_MAX = -122.75, -121.30

# Data Loading

def read_parquet_head(path: Path, nrows=None, columns=None) -> pd.DataFrame:

    # Fast Method: use pyarrow to avoid loading the entire file
    # Slow Method: pandas.read_parquet

    try:
        ds_obj = ds.dataset(str(path), format="parquet")
        if nrows is None:
            table = ds_obj.to_table(columns=columns)
        else:
            table = ds_obj.scanner(columns=columns).head(nrows)
        return table.to_pandas()

    except Exception:
        df = pd.read_parquet(path, columns=columns)
        return df.head(nrows) if nrows else df
    
# H3 Helpers
    
# def in_bounds(cell_str: str) -> bool:
#     lat, lon = h3.cell_to_latlng(cell_str)
#     return (LAT_MIN <= lat <= LAT_MAX and
#             LON_MIN <= lon <= LON_MAX)

def hex_boundary_geojson(cell_str: str) -> list:
    raw = h3.cell_to_boundary(cell_str)
    coords = [[lon, lat] for lat, lon in raw]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords

# Per-User, Time-Ordered Hex Sequences
    
def build_user_sequences_with_time(df: pd.DataFrame, res: int) -> dict[str, list[tuple[str, float]]]:
    """
    For each user, sort pings by timestamp, map to H3, and collapse
    consecutive duplicates. Returns {user_id: [(cell, timestamp), ...]}.
    Only keeps cell inside bounding box
    """
    df = df.sort_values([USER_COL, TIME_COL]).reset_index(drop=True)

    lats = df[LAT_COL].to_numpy()
    lons = df[LON_COL].to_numpy()
    cells = np.array([
        h3.latlng_to_cell(float(la), float(lo), res)
        if not (math.isnan(la) or math.isnan(lo)) else None
        for la, lo in zip(lats, lons)
    ])
    df["cell"] = cells
    df = df[df["cell"].notna()].copy()

    # unique_cells  = df["cell"].unique()
    # bounds_lookup = {c: in_bounds(c) for c in unique_cells}   # one call per unique cell
    # df = df[df["cell"].map(bounds_lookup)]

    df["prev_cell"] = df.groupby(USER_COL)["cell"].shift(1)
    df = df[df["cell"] != df["prev_cell"]]

    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")

    sequences: dict[str, list[tuple[str, float]]] = {}
    for uid, group in df.groupby(USER_COL):
        sequences[uid] = list(zip(group["cell"], group[TIME_COL]))

    return sequences

# Extract all A->B Trips

def find_all_ab_trips(sequences: dict[str, list[tuple[str, float]]], time_threshold: float, max_length: int = MAX_TRIP_LENGTH) -> dict[tuple[str, str], list[list[str]]]:
    """
    For every user trajectory, iterate all ordered point pairs (pointA, pointB)
    If timeB - timeA <= time_threshold and hexA != hexB, record the hex
    subsequence as a trip. Returns {(hexA, hexB): [trip, trip,...]}
    """

    all_trips: dict[tuple[str, str], list[list[str]]] = defaultdict(list)

    for uid, seq in sequences.items():
        N = len(seq)
        for i in range(N):
            hexA, timeA = seq[i]
            for j in range(i + 1, N):
                hexB, timeB = seq[j]
                elapsed = timeB - timeA
                if elapsed > time_threshold:
                    break       # seq is time sorted, so no point checking further
                if hexA == hexB:
                    continue
                if haversine_km(hexA, hexB) < MIN_KM_DIST:
                    continue
                trip = [cell for cell, _ in seq[i : j + 1]]
                if len(trip) <= max_length and len(trip) == len(set(trip)):
                    if len(all_trips[(hexA, hexB)]) < MAX_TRIPS_PER_PAIR:
                        all_trips[(hexA, hexB)].append(trip)

    return dict(all_trips)

# Compute P(H) = N_H / N for all (hexA, hexB) pairs

def compute_all_hex_probabilities(all_trips: dict[tuple[str, str], list[list[str]]]) -> dict[tuple[str, str], dict[str, float]]:
    """ 
    For each (hexA, hexB):
    N = total trips for that pair
    N_H = trips that pass through hex H
    P(H) = N_H / N
    Returns {(hexA, hexB): {cell: P(H)}}
    """

    result: dict[tuple[str, str], dict[str, float]] = {}
    for (hexA, hexB), trips in all_trips.items():
        N = len(trips)
        visit_counts: dict[str, int] = defaultdict(int)
        for trip in trips:
            for cell in set(trip):
                visit_counts[cell] += 1
        result[ (hexA, hexB) ] = {cell: cnt / N for cell, cnt in visit_counts.items()}

    return result

# Most Likely Observed Path
    
def most_common_path(trips: list[list[str]]) -> list[str]:
    """
    Treat each trip as a tuple and count exact occurrences
    The most frequent tuple is the most liekly observed path
    """

    if not trips:
        return []
    path_counts = Counter(tuple(trip) for trip in trips)
    best_path, count = path_counts.most_common(1)[0]
    print(f"    Most common path: {count}/{len(trips)} trips ({count/len(trips):.1%}) ")
    
    return list(best_path)

# Output File
def build_trip_summary_csv(
    all_trips: dict[tuple[str, str], list[list[str]]],
    out_path: Path = Path("trip_summary.csv"),
):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar=' ')
        writer.writerow(["Start Hexagon", " End Hexagon", " Trip Count"])

        for (hexA, hexB), trips in sorted(all_trips.items(), key=lambda x: len(x[1]), reverse=True):
            N = len(trips)
            visit_counts: dict[str, int] = defaultdict(int)
            for trip in trips:
                for cell in trip[1:-1]:
                    visit_counts[cell] += 1

            sorted_hexes = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
            row = [hexA, hexB, N]
            for cell, count in sorted_hexes:
                row.append(f"({cell}:{count})")
            writer.writerow(row)

    print(f"Saved trip summary → {out_path.resolve()}  ({len(all_trips):,} rows)")

# Folium Map
ROUTE_COLORS = ["#0057e7", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

def build_map(
        hex_probs: dict[str, float],
        top_paths: list[tuple[list[str], int]],
        hex_a: str,
        hex_b: str,
        center: tuple[float, float],
        N: int,
) -> folium.Map:
    
    m = folium.Map(location=list(center), zoom_start=ZOOM_START, tiles="CartoDB Positron")

    if not top_paths:
        return m
    
    all_path_cells = {c for path, _ in top_paths for c in path}
    path_probs = {c: hex_probs[c] for c in all_path_cells if c in hex_probs}

    if path_probs:
        min_p, max_p = min(path_probs.values()), max(path_probs.values())
        colormap = cm.linear.YlOrRd_09.scale(min_p, max_p)
        colormap.caption = "P(H) - visit probability on A -> B"
        colormap.add_to(m)
    else:
        colormap = None

    # Draw each hex in the path
    for cell_str in all_path_cells:
        coords = hex_boundary_geojson(cell_str)
        latlon = [(lat, lon) for lon, lat in coords]
        prob = path_probs.get(cell_str, 0.0)
        folium.Polygon(
            locations = latlon,
            color = "#333333",
            weight = 0.8,
            fill = True,
            fill_color = colormap(prob) if colormap else "#3388ff",
            fill_opacity = 0.65,
            tooltip = f"<b>{cell_str}</b><br>P(H) = {prob:.4f}",
        ).add_to(m)
        
    # Single polyline through hex centers
    for rank, (path, count) in enumerate(top_paths):
        folium.PolyLine(
            locations = [h3.cell_to_latlng(c) for c in path],
            color = ROUTE_COLORS[rank % len(ROUTE_COLORS)],
            weight = 5,
            opacity = 0.80,
            tooltip = f"Route #{rank+1}: {count}/{N} trips ({count/N:.1%})",
        ).add_to(m)

    folium.Marker(h3.cell_to_latlng(hex_a), tooltip=f"A: {hex_a}",
                    icon = folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(h3.cell_to_latlng(hex_b), tooltip=f"B: {hex_b}",
                    icon = folium.Icon(color="red", icon="stop")).add_to(m)
    
    return m

def describe_trips(trips: list[list[str]], hex_probs: dict[str, float], top_k: int = 5):
    """
    Print a breakdown of distinct routes and their frequencies.
    """
    N = len(trips)
    path_counts = Counter(tuple(trip) for trip in trips)

    print(f"\n  Total distinct routes: {len(path_counts)}")
    print(f"  Top {top_k} routes by frequency:\n")

    for rank, (path_tuple, count) in enumerate(path_counts.most_common(top_k), 1):
        share = count / N
        avg_prob = sum(hex_probs.get(c, 0) for c in path_tuple) / len(path_tuple)
        print(f"  Route #{rank}:")
        print(f"    Frequency : {count}/{N} trips ({share:.1%})")
        print(f"    Length    : {len(path_tuple)} hexes")
        print(f"    Avg P(H)  : {avg_prob:.4f}")
        print(f"    Path      : {' → '.join(path_tuple)}")
        print()

def haversine_km(cell_a: str, cell_b: str) -> float:
    lat1, lon1 = h3.cell_to_latlng(cell_a)
    lat2, lon2 = h3.cell_to_latlng(cell_b)
    R = 6371
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

def main():
    t0 = time.time()
    #hex_a, hex_b = HEX_A_STR, HEX_B_STR

    # Load Data
    print("Loading parquet...")
    use_cols = [USER_COL, TIME_COL, LAT_COL, LON_COL]
    df = read_parquet_head(FILE_PATH, nrows=NUM_ROWS, columns=use_cols)
    print(f"Rows loaded: {len(df):,}")

    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    # df = df.loc[
    #     df[LAT_COL].between(LAT_MIN, LAT_MAX) & 
    #     df[LON_COL].between(LON_MIN, LON_MAX)
    # ].copy()
    # print(f"    After bbox: {len(df):,}")

    # Sequences
    print("Building per-user sequence...")
    sequences = build_user_sequences_with_time(df, RES)
    print(f"    Unique users: {len(sequences):,}")
    
    # Trips
    print(f"Finding all A->B trips (time_threshold={TIME_THRESHOLD}s)...")
    all_trips = find_all_ab_trips(sequences, TIME_THRESHOLD)
    total_trips = sum(len(v) for v in all_trips.values())
    print(f"    Unique (hexA, hexB) pairs: {len(all_trips):,}")
    print(f"    Total trips across pairs : {total_trips:,}")

    # CSV Output
    print("Writing trip summary CSV...")
    summary_df = build_trip_summary_csv(all_trips, Path("trip_summary.csv"))

    # P(H) for every pair
    print("Computing P(H) for all (hexA, hexB) pairs...")
    all_hex_probs = compute_all_hex_probabilities(all_trips)
    print(f" Done - {len(all_hex_probs):,} distributions completed")

    # Visualize fixed (HEX_A, HEX_B) pair
    key = max(
        (k for k in all_trips
            if haversine_km(k[0], k[1]) >= MIN_KM_DIST
            and (sum(len(t) for t in all_trips[k]) / len(all_trips[k])) >= 4),
        key=lambda k: len(all_trips[k])
    )
    
    hex_a, hex_b = key
    trips = all_trips[key]
    hex_probs = all_hex_probs[key]
    N = len(trips)
    print(f"\n Fixed pair ({hex_a} -> {hex_b}): N = {N} trips")

    if N == 0:
        print("No trips found for this pair.")
        return
    
    print(f"    Unique hexes with P(H) > 0: {len(hex_probs)}")
    top5 = sorted(hex_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("     Top 5 by P(H):")
    for cell, p in top5:
        print(f"    {cell} P={p:.4f}")

    describe_trips(trips, hex_probs)

    TOP_K = 5
    path_counts = Counter(tuple(t) for t in trips)
    top_paths = [(list(p), cnt) for p, cnt in path_counts.most_common(TOP_K)]

    lat_a, lon_a = h3.cell_to_latlng(hex_a)
    lat_b, lon_b = h3.cell_to_latlng(hex_b)
    center = ((lat_a + lat_b) / 2, (lon_a + lon_b) / 2)

    m = build_map(hex_probs, top_paths, hex_a, hex_b, center, N)
    m.save(str(OUT_HTML))
    print(f"\nsaved: {OUT_HTML.resolve()}")
    print(f"Total time: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
