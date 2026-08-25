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
   - `h3_conversion_rate.py`, `h3_real_conversion.py`, `h3_playground.py` —
     benchmark and sanity-check the lat/lon → H3 conversion step on synthetic
     and real data.

2. **Trip reconstruction & pair mining** — per-user point sequences are
   turned into trips, filtered (time gap threshold, minimum trip distance,
   min/max unique hexes per user, minimum path length), and aggregated into
   "interesting" A→B hex pairs — pairs with enough repeated trips to be worth
   modeling.
   - `interesting_pairs_batch.py` — main batch pipeline.
   - `interesting_pairs_batch_control.py` — control/baseline variant for
     comparison.
   - `interesting_pairs_batch_experiments.py` — parameter-sweep harness; see
     `interesting_pairs_batch_experiments_all_files_combined_5m_to_70m/` for
     results (rows/time, thresholds/time) as data scales from 5M to 70M rows.

3. **Visualization** — interactive Folium maps for inspecting density and
   specific routes (outputs are gitignored; regenerate locally).
   - `heatmap.py` — hexagon density heatmap.
   - `probability_map.py` / `probability_map_v2.py` (+ `_control`) —
     probability-of-visit maps.
   - `poster_hex_example_map.py` — figure-quality example map for
     presentations/posters.

## Data

Raw trip data (`*.parquet`, `*.csv`) and generated map HTML are gitignored —
they're large and regeneratable/non-source. Point the scripts at your local
`data_files/` directory to reproduce results.

## Status

Early-stage: pipeline mines and visualizes interesting A→B pairs. Route
*prediction* (the ultimate goal) has not been built yet — current work is
focused on figuring out how much data/what thresholds are needed to make
that feasible.
