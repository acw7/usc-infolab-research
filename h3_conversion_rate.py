# This script benchmarks synthetic latitude/longitude conversion into H3 cells.
# It generates fake LA-area points and reports per-batch conversion speed.
import pandas as pd
import numpy as np
import h3
import time
from collections import Counter

# Hard-Coded Data bc Node5 was down
N_ROWS = 10_000_000
BATCH_SIZE = 1_000_000
H3_RESOLUTION = 5

# Rough Bounds for LA Area
MIN_LAT, MAX_LAT = 34.02, 34.10
MIN_LON, MAX_LON = -118.30, -118.20

np.random.seed(42)

# Generate N_ROWS of random (lat, lon) coordinates
# Combine that into a pandas DF
df = pd.DataFrame({
    "latitude": np.random.uniform(MIN_LAT, MAX_LAT, N_ROWS),
    "longitude": np.random.uniform(MIN_LON, MAX_LON, N_ROWS),
})

print("Generated synthetic GPS Data:")
print(df.head(), "\n")

# H3 Conversion
hex_counts = Counter()

total_rows = 0
total_time = 0.0
batch_num = 0

# Extract lat and lon columns as NumPy arrays
lats = df["latitude"].to_numpy()
lons = df["longitude"].to_numpy()
print("Starting H3 conversion...\n")

# Iterate over data in steps of BATCH_SIZE
for start in range(0, N_ROWS, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N_ROWS)
    batch_num += 1

    # Slice out current batch
    batch_lats = lats[start:end]
    batch_lons = lons[start:end]
    n = len(batch_lats)

    # Time the conversion and convert GPS points into H3 hexes
    t0 = time.perf_counter()
    hexes = [
        h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
        for lat, lon in zip(batch_lats, batch_lons)
    ]

    dt = time.perf_counter() - t0
    for h in hexes:
        hex_counts[h] += 1

    # Compute conversion rate and update totals
    rate = n / dt if dt > 0 else float("inf")
    total_rows += n
    total_time += dt

    # Per-batch summary
    print(
        f"Batch {batch_num:02d}: "
        f"{n} rows | {dt:.3f} s | "
        f"{int(rate):,} conversions/sec"
    )

# Summary
average_rate = total_rows / total_time
print("\n=== SUMMARY ===")
print(f"Total rows processed: {total_rows}")
print(f"Total conversion time: {total_time:.3f} s")
print(f"Average conversion rate: {int(average_rate):,} conversions/sec")
print(f"Unique hexagons: {len(hex_counts)}")

print("\nTop 10 most frequent hexes:")
for h, c in hex_counts.most_common(10):
    print(h, c)
