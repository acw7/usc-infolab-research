# USC InfoLab Research — Predicted Routes Between H3 Hexagons

Research into predicting likely travel routes between arbitrary pairs of H3
hexagons (A → B), using large-scale device GPS trip data. The long-term goal
is a model that, given any origin/destination hexagon pair, predicts the
probable path between them. Getting there first requires characterizing what
data volume and thresholds are needed to make that prediction tractable —
which is the current focus of this repo.

## Pipeline

1. **Ingest & convert** — daily parquet files of raw GPS pings (lat/lon,
   timestamp, user/CAID) are scanned under a shared row budget and converted
   to H3 cells (res 9).

2. **Trip reconstruction & pair mining** — per-user point sequences are
   turned into trips, filtered (time gap threshold, minimum trip distance,
   min/max unique hexes per user, minimum path length), and aggregated into
   "interesting" A→B hex pairs — pairs with enough repeated trips to be worth
   modeling.
   - `interesting_pairs_batch.py` — main batch pipeline.
   - `interesting_pairs_batch_experiments.py` — parameter-sweep harness; see
     `interesting_pairs_batch_experiments_all_files_combined_5m_to_70m/` for
     results (rows/time, thresholds/time) as data scales from 5M to 70M rows.

3. **Visualization** — interactive Folium maps for inspecting density and
   specific routes (outputs are gitignored; regenerate locally).
   - `heatmap.py` — hexagon density heatmap.
   - `probability_map_v2.py` — probability-of-visit maps.
   - `poster_hex_example_map.py` — figure-quality example map for
     presentations/posters.

Removed scripts (H3 conversion benchmarks, `_control` baselines,
`probability_map.py` v1) live in git history at commit `8fc5f1e`.

## Data

Raw trip data (`*.parquet`, `*.csv`) and generated map HTML are gitignored —
they're large and regeneratable/non-source. Point the scripts at your local
`data_files/` directory to reproduce results.

## Status

Early-stage: pipeline mines and visualizes interesting A→B pairs. Route
*prediction* (the ultimate goal) has not been built yet — current work is
focused on figuring out how much data/what thresholds are needed to make
that feasible.

## Route API (MVP)

`route_api.py` predicts the most likely route between two lat/lon points.
It builds a hex-to-hex transition graph from all user ping sequences (edge
weight = `-log P(next hex | current hex)`), then runs Dijkstra, so it works
even for A→B pairs never observed as a single trip.

```bash
python3 route_api.py build data_files/january graph.pkl --rows-per-file 5000000
python3 route_api.py serve graph.pkl --host <lab-internal-ip> --port 8000
curl "http://<host>:8000/route?lat_a=37.7823&lon_a=-122.4083&lat_b=37.3853&lon_b=-122.0311"
python3 route_api.py demo   # self-check
```

Response: `{"probability": float, "path": [{"cell", "lat", "lon"}, ...]}`.
400 = bad input, 404 = no route. Internal network only: no auth.
