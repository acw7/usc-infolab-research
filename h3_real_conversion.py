# This script benchmarks converting rows from the original parquet data into H3 cells.
# It prints load/conversion timing and the most frequent resulting hexagons.
import pandas as pd
import numpy as np
import h3.api.basic_int as h3
import time
from collections import Counter
import pyarrow.parquet as pq

path = "data_files/2019_01_01.parquet"
pf = pq.ParquetFile(path)
total_rows = pf.metadata.num_rows
print("Total rows:", total_rows)

# Read data

NUM_ROWS = 100_000_000

parse_start = time.time()

df = pd.read_parquet(path)
df_head = df.head(NUM_ROWS).copy()

print("Loaded rows:", len(df_head))
print(df_head.columns)

parse_end = time.time()

res = 13
lat = df_head["latitude"].to_numpy()
lon = df_head["longitude"].to_numpy()

start = time.time()

h3_list = []

for a, b in zip(lat, lon):
    if np.isnan(a) or np.isnan(b):
        h3_list.append(0)
    else:
        h3_list.append(h3.latlng_to_cell(float(a), float(b), res))

end = time.time()
elapsed = end - start
parse_elapsed = parse_end - parse_start

df_head[f"h3_r{res}"] = np.array(h3_list, dtype=np.uint64)
hex_counts = df_head[f"h3_r{res}"].value_counts().head(10)

print(f"Mapped {len(h3_list)} rows to H3 resolution {res}")
print("Most frequent hexagon counts")
print(hex_counts)

print(f"time: {elapsed:.2f}")
print(f"parse time: {parse_elapsed:.2f}")
print("conversion rate:", len(h3_list) / elapsed, "conversion/sec")

# Data Parsing Logs
# 1. 100 rows = 68.46s
# 2. 1000 rows = 71.07s
# 3. 2500 rows = 67.61s
# 4. 10000 rows = 69.32s
# 5. 20000 rows = 72.82s
# 6. 30000 rows = 77.32s
# 7. 50000 rows = 66.98s
# 8. 100000 rows = 65.40s
# 9. 250000 rows = 69.47s
# 10. 1M rows = 73.17s
# 11. 5M rows = 72.53s
# 12. 7.5M rows = 74.33s; 17s to store data (res = 13)
# 13. 7.5M rows = 73.05s; 16s to store data (res = 9)
# 14. 10M rows = 77.65s 21s to store data (res = 9)
# 15. 25M rows = 77.66s; 53s to store data (res = 9)
# 16. 25M rows = 75.56s; 57s to store data (res = 13)
