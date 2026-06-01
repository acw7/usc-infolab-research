# This script builds a Folium heatmap-style H3 polygon map from parquet GPS points.
# It reads a configurable row subset, aggregates H3 counts, and writes an HTML map.
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import folium
import branca.colormap as cm
import h3
import h3.api.basic_int as h3_int
import h3 as h3_str_api
import pyarrow.dataset as ds

# Data

FILE_PATH = Path("data_files/2019_01_01.parquet")
NUM_ROWS = None
LAT_COL = "latitude"
LON_COL = "longitude"
RES = 9
TOP_K = None
OUT_HTML = Path("heatmap.html")
ZOOM_START = 11
COLORMAP_NAME = "YlOrRd"
LINE_COLOR = "#222222"
FILL_OPACITY = 0.65

# Helper Functions

def read_parquet_head(path: Path, nrows=None, columns=None) -> pd.DataFrame:

    # Fast Method: use pyarrow to avoid loading the entire file
    # Slow Method: pandas.read_parquet

    try:
        ds_obj = ds.dataset(str(path), format="parquet")
        if nrows is None:
            table = ds_obj.to_table(columns=columns)
        else:
            table = ds_obj.scanner(columns=columns).head(nrows)
        df = table.to_pandas()
        return df

    except Exception:
        df = pd.read_parquet(path, columns=columns)
        if nrows is not None:
            df = df.head(nrows)

        return df

def map_points_to_h3(lat_arr: np.ndarray, lon_arr: np.ndarray, res: int) -> np.ndarray:

    # Convert arrays of lat/lon floats to H3 indices
    
    n = len(lat_arr)
    out = np.empty(n, dtype=np.uint64)

    for i, (a, b) in enumerate(zip(lat_arr, lon_arr)):
        if np.isnan(a) or np.isnan(b):
            out[i] = 0
            continue
        cell = h3.latlng_to_cell(float(a), float(b), res)

        if isinstance(cell, str):
            out[i] = int(cell, 16) if cell else 0
        else:
            out[i] = int(cell)
    return out

def aggregate_counts(h3_array: np.ndarray, drop_zero=True) -> dict:

    # Return dict mapping h3_int -> count

    s = pd.Series(h3_array)
    counts = s.value_counts().to_dict()
    if drop_zero and 0 in counts:
        counts.pop(0)
    
    return {int(k): int(v) for k, v in counts.items()}

def h3_index_to_geojson_feature(h3_idx: int, count: int) -> dict:

    # Converts a single integer H3 index to a GeoJSON Feature dict with {'h3', 'count'}

    boundary_latlon = None
    try:
        hex_str = h3_int.int_to_str(h3_idx)
        boundary_latlon = h3_str_api.h3_to_geo_boundary(hex_str, geo_json=True)

    except Exception:

        if hasattr(h3_int, "cell_to_boundary"):

            try:
                b = h3_int.cell_to_boundary(h3_idx, geo_json=True)
                boundary_latlon = [ [float(lat), float(lon)] for lat, lon in b]
            except Exception:
                b = h3_int.cell_to_boundary(h3_idx)
                boundary_latlon = [ [float(lat), float(lon)] for lat, lon in b]

        elif hasattr(h3_int, "h3_to_geo_boundary"):
            b = h3_int.h3_to_geo_boundary(h3_idx, geo_json=True)
            boundary_latlon = [ [float(lat), float(lon)] for lat, lon in b]

    if not boundary_latlon:
        raise RuntimeError("Could not obtain H3 boundary for index: {}".format(h3_idx))
    
    # Ensure polygon closure
    if boundary_latlon[0] != boundary_latlon[-1]:
        boundary_latlon.append(boundary_latlon[0])

    # Convert to GeoJSON coordinate order: [lon, lat]
    coords = [[ [lon, lat] for lat, lon in boundary_latlon]]
    feature = {
        "type": "Feature",
        "properties": {"h3": int(h3_idx), "count": int(count)},
        "geometry": {"type": "Polygon", "coordinates": coords}
    }
    return feature

def build_geojson_feature_collection(counts_dict: dict, top_k=None) -> dict:
    
    # Build a GeoJSON FeatureCollection of hex polygons

    items = sorted(counts_dict.items(), key=lambda kv: kv[1], reverse=True)
    if top_k and top_k > 0:
        items = items[:top_k]
    
    features = []
    for h3_idx, cnt in items:
        feat = h3_index_to_geojson_feature(int(h3_idx), int(cnt))
        features.append(feat)
    
    return {"type": "FeatureCollection", "features": features}

def create_choropleth_map(geojson_fc: dict, counts_df: pd.DataFrame, center: tuple, zoom_start=11):
    
    # Create a folium map with built-in Choropleth

    m = folium.Map(location=[center[0], center[1]], zoom_start=zoom_start, tiles="CartoDB Positron")
    folium.Choropleth(
        geo_data = geojson_fc,
        data = counts_df,
        columns=["h3", "count"],
        key_on = "feature.properties.h3",
        fill_color = COLORMAP_NAME,
        fill_opacity = FILL_OPACITY,
        line_opacity = 0.8,
        legend_name = "Points per hex"
    ).add_to(m)

    def style_fn(feature):
        return {"color": LINE_COLOR, "weight": 1, "fillOpacity": 0}
    
    folium.GeoJson(geojson_fc, style_function=style_fn, name="Hex outlines").add_to(m)
    folium.LayerControl().add_to(m)
    return m

def main():

    # Bounding-Rectangle Filter (Bay Area)
    LAT_MIN = 36.80
    LAT_MAX = 38.40
    LON_MIN = -122.75
    LON_MAX = -121.30

    t0 = time.time()
    use_cols = [LAT_COL, LON_COL]
    print("Reading input (fast partial when possible)...")
    df = read_parquet_head(FILE_PATH, nrows=NUM_ROWS, columns=use_cols)
    print("Loaded rows:", len(df))

    df[LAT_COL] = df[LAT_COL].astype(float)
    df[LON_COL] = df[LON_COL].astype(float)

    in_bound_mask = (
        (df[LAT_COL] >= LAT_MIN) &
        (df[LAT_COL] <= LAT_MAX) &
        (df[LON_COL] >= LON_MIN) &
        (df[LON_COL] <= LON_MAX)
    )

    df = df.loc[in_bound_mask].copy()
    lat = df[LAT_COL].to_numpy()
    lon = df[LON_COL].to_numpy()

    print("Mapping points to H3...")
    t_map0 = time.time()
    h3_arr = map_points_to_h3(lat, lon, RES)
    t_map1 = time.time()
    print(f"H3 mapping elapsed: {t_map1 - t_map0:.2f} s")

    df[f"h3_r{RES}"] = h3_arr

    print("Aggregating counts per hex...")
    counts = aggregate_counts(h3_arr, drop_zero=True)
    print("Unique hex cells (after drop zero):", len(counts))

    # Small DataFrame for Choropleth
    counts_df = pd.DataFrame([(int(k), int(v)) for k, v in counts.items()], columns=["h3", "count"])

    # Create GeoJSON features
    print("Building GeoJSON for top_k:", TOP_K)
    geojson_fc = build_geojson_feature_collection(counts, top_k=TOP_K)

    # Map center: mean of valid coordinates
    valid_mask = (~np.isnan(lat) & (~np.isnan(lon)))
    if valid_mask.any():
        center_lat = float(np.mean(lat[valid_mask]))
        center_lon = float(np.mean(lon[valid_mask]))
    else:
        center_lat, center_lon = 0.0, 0.0

    print("Creating folium map...")
    m = create_choropleth_map(geojson_fc, counts_df, center=(center_lat, center_lon), zoom_start=ZOOM_START)

    folium.Rectangle(
        bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
        color="blue",
        weight=2,
        fill=False,
        opacity=0.4
    ).add_to(m)

    m.add_child(folium.LatLngPopup())

    print("Saving map to:", OUT_HTML)
    m.save(str(OUT_HTML))

    total = time.time() - t0
    print(f"Done. Total time elapsed: {total:.2f} s")
    print(f"Map file: {OUT_HTML.resolve()}")

if __name__ == "__main__":
    main()
