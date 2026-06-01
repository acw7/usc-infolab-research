# This script creates a poster-friendly Folium example of GPS points assigned to H3 hexes.
# It focuses on a chosen LA area and writes a small explanatory HTML map.
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import folium
import h3
import pandas as pd
import pyarrow.dataset as ds

FILE_PATH = Path("data_files/2019_01_01.parquet")
LAT_COL = "latitude"
LON_COL = "longitude"
H3_RESOLUTION = 8
DEFAULT_OUTPUT = Path("poster_hex_example_map.html")

AREAS = {
    "santa_monica": {
        "label": "Santa Monica",
        "bbox": (33.995, 34.025, -118.505, -118.465),
        "center": (34.014, -118.491),
        "anchor_cell": "8829a19aa3fffff",
        "zoom_start": 14,
    },
    "downtown": {
        "label": "Downtown LA",
        "bbox": (34.035, 34.060, -118.275, -118.235),
        "center": (34.047, -118.255),
        "anchor_cell": "8829a1d62dfffff",
        "zoom_start": 14,
    },
    "la_405": {
        "label": "405 Corridor",
        "bbox": (34.045, 34.085, -118.475, -118.425),
        "center": (34.064, -118.446),
        "anchor_cell": "8829a199d1fffff",
        "zoom_start": 13,
    },
}

NEUTRAL_HEX_FILL = "#fff7b3"
NEUTRAL_HEX_LINE = "#c7b84a"
POINT_COLOR = "#dc2626"
HIGHLIGHT_COLORS = ["#ffe066", "#ffd43b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a poster-friendly Folium map that shows GPS points mapped into H3 hexes."
    )
    parser.add_argument(
        "--area",
        choices=sorted(AREAS),
        default="santa_monica",
        help="LA area to center in the example figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="HTML file to write.",
    )
    parser.add_argument(
        "--cells-to-show",
        type=int,
        default=5,
        help="Number of nearby hexes to display.",
    )
    return parser.parse_args()


def load_points_in_bbox(path: Path, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    lat_min, lat_max, lon_min, lon_max = bbox
    dataset = ds.dataset(str(path), format="parquet")
    lat = ds.field(LAT_COL)
    lon = ds.field(LON_COL)
    bbox_filter = (
        (lat >= lat_min)
        & (lat <= lat_max)
        & (lon >= lon_min)
        & (lon <= lon_max)
    )
    table = dataset.to_table(columns=[LAT_COL, LON_COL], filter=bbox_filter)
    frame = table.to_pandas()
    frame[LAT_COL] = pd.to_numeric(frame[LAT_COL], errors="coerce")
    frame[LON_COL] = pd.to_numeric(frame[LON_COL], errors="coerce")
    frame = frame.dropna(subset=[LAT_COL, LON_COL]).reset_index(drop=True)
    frame["cell"] = [
        h3.latlng_to_cell(float(lat_value), float(lon_value), H3_RESOLUTION)
        for lat_value, lon_value in zip(frame[LAT_COL], frame[LON_COL])
    ]
    return frame


def select_focus_cells(
    frame: pd.DataFrame,
    anchor_cell: str,
    cells_to_show: int,
) -> tuple[list[str], list[str], Counter]:
    counts = Counter(frame["cell"])
    first_ring = [
        cell for cell in h3.grid_disk(anchor_cell, 1) if counts.get(cell, 0) > 0
    ]
    first_ring_sorted = sorted(
        first_ring,
        key=lambda cell: (cell != anchor_cell, -counts[cell]),
    )

    highlight_cells = first_ring_sorted[:2]
    selected_cells: list[str] = []
    for cell in highlight_cells + first_ring_sorted:
        if cell not in selected_cells:
            selected_cells.append(cell)

    if len(selected_cells) < cells_to_show:
        for cell, _ in counts.most_common():
            if cell not in selected_cells:
                selected_cells.append(cell)
            if len(selected_cells) >= cells_to_show:
                break

    return selected_cells[:cells_to_show], highlight_cells, counts


def sample_points(frame: pd.DataFrame, selected_cells: list[str], highlight_cells: list[str]) -> pd.DataFrame:
    samples = []
    for index, cell in enumerate(selected_cells):
        points_in_cell = frame.loc[frame["cell"] == cell, [LAT_COL, LON_COL]].copy()
        n_samples = 16
        sampled = points_in_cell.sample(
            n=min(n_samples, len(points_in_cell)),
            random_state=17 + index,
        ).copy()
        sampled["cell"] = cell
        samples.append(sampled)

    return pd.concat(samples, ignore_index=True)


def cell_boundary(cell: str) -> list[tuple[float, float]]:
    return [(lat, lon) for lat, lon in h3.cell_to_boundary(cell)]


def add_legend(map_obj: folium.Map, label: str) -> None:
    legend = f"""
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        width: 240px;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #94a3b8;
        border-radius: 8px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.18);
        padding: 10px 12px;
        font-family: Arial, sans-serif;
        font-size: 12px;
        color: #0f172a;
    ">
        <div style="font-weight: 700; margin-bottom: 6px;">{label}</div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block;width:12px;height:12px;background:{HIGHLIGHT_COLORS[0]};border:1px solid {NEUTRAL_HEX_LINE};margin-right:8px;"></span>
            highlighted hex example
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block;width:12px;height:12px;background:{HIGHLIGHT_COLORS[1]};border:1px solid {NEUTRAL_HEX_LINE};margin-right:8px;"></span>
            second assigned hex
        </div>
        <div>
            <span style="display:inline-block;width:12px;height:12px;background:{NEUTRAL_HEX_FILL};border:1px solid {NEUTRAL_HEX_LINE};margin-right:8px;"></span>
            nearby hexes for context
        </div>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend))


def build_map(
    area_name: str,
    selected_cells: list[str],
    highlight_cells: list[str],
    sampled_points: pd.DataFrame,
) -> folium.Map:
    area = AREAS[area_name]
    lat_min, lat_max, lon_min, lon_max = area["bbox"]
    fmap = folium.Map(
        location=list(area["center"]),
        zoom_start=area["zoom_start"],
        tiles="OpenStreetMap",
        control_scale=True,
    )

    highlight_styles = {
        cell: {
            "fill": HIGHLIGHT_COLORS[index],
            "line": NEUTRAL_HEX_LINE,
        }
        for index, cell in enumerate(highlight_cells)
    }

    for cell in selected_cells:
        style = highlight_styles.get(
            cell,
            {"fill": NEUTRAL_HEX_FILL, "line": NEUTRAL_HEX_LINE},
        )
        folium.Polygon(
            locations=cell_boundary(cell),
            color=style["line"],
            weight=3 if cell in highlight_cells else 2,
            fill=True,
            fill_color=style["fill"],
            fill_opacity=0.72 if cell in highlight_cells else 0.4,
        ).add_to(fmap)

    for row in sampled_points.itertuples(index=False):
        style = highlight_styles.get(
            row.cell,
            {"fill": POINT_COLOR, "line": POINT_COLOR},
        )
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=5,
            color=style["line"],
            weight=1,
            fill=True,
            fill_color=POINT_COLOR,
            fill_opacity=0.95,
        ).add_to(fmap)

    fmap.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])
    add_legend(fmap, f"{area['label']}: GPS points assigned to H3 hexes")
    return fmap


def main() -> None:
    args = parse_args()
    area = AREAS[args.area]

    print(f"Loading points for {area['label']}...")
    frame = load_points_in_bbox(FILE_PATH, area["bbox"])
    print(f"Rows in bbox: {len(frame):,}")

    selected_cells, highlight_cells, counts = select_focus_cells(
        frame,
        area["anchor_cell"],
        cells_to_show=args.cells_to_show,
    )
    sampled_points = sample_points(frame, selected_cells, highlight_cells)

    print("Selected cells:")
    for cell in selected_cells:
        marker = "*" if cell in highlight_cells else "-"
        print(f"  {marker} {cell} ({counts[cell]:,} points)")

    print(f"Sampled GPS points: {len(sampled_points)}")
    fmap = build_map(args.area, selected_cells, highlight_cells, sampled_points)
    fmap.save(str(args.output))
    print(f"Saved map to {args.output.resolve()}")


if __name__ == "__main__":
    main()
