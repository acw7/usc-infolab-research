# This script analyzes one fixed H3 A-to-B pair and maps visit probabilities along routes.
# It builds per-user sequences from parquet GPS data and writes an HTML path map.
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

# Data Config
FILE_PATH = Path("data_files/2019_01_01.parquet")
NUM_ROWS = None
LAT_COL = "latitude"
LON_COL = "longitude"
USER_COL = "caid"
TIME_COL = "utc_timestamp"

RES = 9
MAX_TRIP_LENGTH = 100
HEX_A_STR = "89283095e57ffff"
HEX_B_STR = "89283082bcfffff"
OUT_HTML = Path("path_probability_map.html")
ZOOM_START = 13

LAT_MIN, LAT_MAX = 36.80, 38.40
LON_MIN, LON_MAX = -122.75, -121.30

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
    
def in_bounds(cell_str: str) -> bool:
    lat, lon = h3.cell_to_latlng(cell_str)
    return (LAT_MIN <= lat <= LAT_MAX and
            LON_MIN <= lon <= LON_MAX)

def hex_boundary_geojson(cell_str: str) -> list:
    raw = h3.cell_to_boundary(cell_str)
    coords = [[lon, lat] for lat, lon in raw]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords

# Per-User, Time-Ordered Hex Sequences
    
def build_user_sequences(df: pd.DataFrame, res: int) -> dict[str, list[str]]:
    """
    For each user, sort pings by timestamp, map to H3, and collapse
    consecutive dupliates. Only keeps ceels inside the bounding box.
    Returns {user_id: [cell_0, cell_1,...]}
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
    unique_cells  = df["cell"].unique()
    bounds_lookup = {c: in_bounds(c) for c in unique_cells}   # one call per unique cell
    df = df[df["cell"].map(bounds_lookup)]

    df["prev_cell"] = df.groupby(USER_COL)["cell"].shift(1)
    df = df[df["cell"] != df["prev_cell"]]

    sequences = (
        df.groupby(USER_COL)["cell"]
        .apply(list)
        .to_dict()
    )
    return sequences

# Extract all A->B Trips

def find_ab_trips(sequences: dict[str, list[str]], hex_a: str, hex_b: str, max_length: int = MAX_TRIP_LENGTH) -> list[list[str]]:
    """
    For every user, find all non-overlapping sub-sequences that start at 
    Hex A and end at Hex B. A single user can contribute multiple trips.
    Trips longer than max_length are discarded as noise
    """

    trips = []
    for uid, seq in sequences.items():
        i = 0
        while i < len(seq):
            if seq[i] == hex_a:
                for j in range(i + 1, len(seq)):
                    if seq[j] == hex_b:
                        trip = seq[i : j + 1]
                        if len(trip) <= max_length:
                            trips.append(trip)
                        i = j
                        break
            i += 1

    return trips

# Compute P(H) = N_H / N

def compute_hex_probabilities(trips: list[list[str]]) -> dict[str, float]:
    """ 
    For N total A -> B trips:
    N_H = number of trips that pass through hex H
    P(H) = N_H / N
    """

    N = len(trips)
    if N == 0:
        return {}
    
    visit_counts: dict[str, int] = defaultdict(int)
    for trip in trips:
        for cell in set(trip):
            visit_counts[cell] += 1

    return {cell: count / N for cell, count in visit_counts.items()}

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

# Folium Map

def add_endpoint_marker(
    fmap: folium.Map,
    cell_str: str,
    label: str,
    color: str,
    tooltip_prefix: str,
) -> None:
    lat, lon = h3.cell_to_latlng(cell_str)
    folium.CircleMarker(
        location=(lat, lon),
        radius=15,
        color=color,
        weight=3,
        fill=True,
        fill_color="#ffffff",
        fill_opacity=0.95,
        opacity=1.0,
    ).add_to(fmap)
    folium.Marker(
        location=(lat, lon),
        tooltip=f"{tooltip_prefix} ({label}): {cell_str}",
        icon=folium.DivIcon(
            icon_size=(30, 30),
            icon_anchor=(15, 15),
            html=(
                f'<div style="width:30px;height:30px;border-radius:50%;'
                f'border:3px solid {color};background:#ffffff;color:{color};'
                "font-weight:700;font-size:16px;line-height:24px;text-align:center;"
                'font-family:Arial,sans-serif;box-shadow:0 0 0 4px rgba(255,255,255,0.9);">'
                f"{label}</div>"
            ),
        ),
    ).add_to(fmap)

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
        
    add_endpoint_marker(m, hex_a, "A", "#1a9850", "Start")
    add_endpoint_marker(m, hex_b, "B", "#d73027", "End")
    
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

def main():
    t0 = time.time()
    hex_a, hex_b = HEX_A_STR, HEX_B_STR

    # Load Data
    print("Loading parquet...")
    use_cols = [USER_COL, TIME_COL, LAT_COL, LON_COL]
    df = read_parquet_head(FILE_PATH, nrows=NUM_ROWS, columns=use_cols)
    print(f"Rows loaded:: {len(df):,}")

    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    df = df.loc[
        df[LAT_COL].between(LAT_MIN, LAT_MAX) & 
        df[LON_COL].between(LON_MIN, LON_MAX)
    ].copy()
    print(f"    After bbox: {len(df):,}")

    # Sequences
    print("Building per-user sequence...")
    sequences = build_user_sequences(df, RES)
    print(f"    Unique users: {len(sequences):,}")
    
    # Trips
    print(f"Extracting trips {hex_a} -> {hex_b}...")
    trips = find_ab_trips(sequences, hex_a, hex_b)
    N = len(trips)
    print(f"     N = {N} trips")
    if N == 0:
        print("No trips found"); return
    
    # P(H)
    hex_probs = compute_hex_probabilities(trips)
    print(f"    Unique hexes with P(H) > 0: {len(hex_probs)}")
    top5 = sorted(hex_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("     Top 5 by P(H):")
    for cell, p in top5:
        print(f"    {cell} -> P={p:.4f}")

    # Most-likely Path
    # Most likely path → now top-k paths
    TOP_K = 5
    path_counts = Counter(tuple(trip) for trip in trips)
    top_paths = [(list(p), cnt) for p, cnt in path_counts.most_common(TOP_K)]

    print(f"  Distinct routes found: {len(path_counts)}")
    for rank, (path, count) in enumerate(top_paths, 1):
        print(f"  Route #{rank}: {count}/{N} trips ({count/N:.1%}), "
              f"{len(path)} hexes")

    # Map
    lat_a, lon_a = h3.cell_to_latlng(hex_a)
    lat_b, lon_b = h3.cell_to_latlng(hex_b)
    center = ((lat_a + lat_b) / 2, (lon_a + lon_b) / 2)

    m = build_map(hex_probs, top_paths, hex_a, hex_b, center, N)
    m.save(str(OUT_HTML))
    print(f"\nSaved: {OUT_HTML.resolve()}")
    print(f"Total time: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
