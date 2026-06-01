# This script is a scratchpad for learning H3 operations and Folium rendering.
# It experiments with trajectories, grid distances, neighbors, and simple hex maps.
import h3
import folium
import pandas as pd
import numpy as np

# Task 1
# lat, lon = 37.7749, -122.4194 # sf
# for res in range(6, 11):
#     print(res, h3.latlng_to_cell(lat, lon, res))

# Summary: latlng_to_cell(lat, lon, res) returns the hexagon ID
# the lat, lon pair resides in with the given resolution

# Task 2
# h = h3.latlng_to_cell(lat, lon, 9)
# neighbors = h3.grid_disk(h, k = 2)
# print(len(neighbors))

# Summary: grid_disk(h,k) returns the number of neighbors to a given hexagon

# Task 3
# p1 = (36.7749, -112.4194)
# p2 = (37.7849, -122.4094)

# h1 = h3.latlng_to_cell(*p1, 9)
# h2 = h3.latlng_to_cell(*p2, 9)
# print(h3.grid_distance(h1, h2))

# Summary: * is the unpacking op
#          grid_distance(h1, h2) returns grid distance between h1 and h2

# Task 4
# lats = np.linspace(37.7749, 37.7849, 20)        # 20 lat values evenly spaced between bounds
# lons = np.linspace(-122.4194, -122.4094, 20)    # 20 long values evenly spaced between bounds
# trajectory = list(zip(lats, lons))

# hex_trajectory = [h3.latlng_to_cell(lat, lon, 9) for lat, lon in trajectory]
# hex_traj_unique = [hex_trajectory[0]]
# for h in hex_trajectory[1:]:
#     if h != hex_traj_unique[-1]:
#         hex_traj_unique.append(h)

# Summary: Simulate hex trajectory by calculating hex grid IDs and filtering
# unique IDs

# Task 5

# Keep first 3 hexes, and everything after index 5
# Removes 3, 4, 5
# observed = hex_traj_unique[:3] + hex_traj_unique[6:]

# start = observed[2]     # Last known hex before the gap
# end = observed[3]       # First known hex after the gap

# # Hexes within 3 hex-steps of last observed location
# candidates = h3.grid_disk(start, k=3)

# # Remove candidates that are too far from the end point to reasonably reach
# plausible = [
#     h for h in candidates
#     if h3.grid_distance(h, end) <= 3
# ]

# Summary: Given a start and end hex with removed data in betwee,
# can we figure out the possible hexes that the person traveled through?

# Task 6
# h9 = h3.latlng_to_cell(lat, lon, 9)
# h7 = h3.cell_to_parent(h9, 7)
# children = h3.cell_to_children(h7, 9)
# print(children)

# Folium
lat, lon = 37.7749, -122.4194 # sf
m = folium.Map(
    location=[lat, lon],    # initial center
    zoom_start=13,          # 13 = neighborhoods
    tiles="OpenStreetMap"   # base map imagery
)

lats = np.linspace(37.7749, 37.7849, 20)
lons = np.linspace(-122.4194, -122.4094, 20)
trajectory = list(zip(lats, lons))
for lat, lon in trajectory:
    folium.CircleMarker (
        location = [lat, lon],
        radius = 3,
        color="blue",
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

res = 9
hex_traj = [h3.latlng_to_cell(lat, lon, res) for lat, lon in trajectory]
hex_traj_unique = [hex_traj[0]]
for h in hex_traj[1:]:
    if h != hex_traj_unique[-1]:
        hex_traj_unique.append(h)

def hex_to_polygon(h):
    boundary = h3.cell_to_boundary(h)
    return [(lat, lon) for lat, lon in boundary]

for h in hex_traj_unique:
    folium.Polygon(
        locations = hex_to_polygon(h),
        color="red",
        weight=2,
        fill=False
    ).add_to(m)

m.save("h3_step5.html")
