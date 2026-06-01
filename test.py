# This tiny script converts two hard-coded coordinates into H3 cells.
# It is a quick sanity check for expected start and end hex IDs.
import h3
lat_a, lon_a = 37.7856, -122.4086
lat_b, lon_b = 37.6164, -122.3805
hex_id_a = h3.latlng_to_cell(lat_a, lon_a, res=9)
hex_id_b = h3.latlng_to_cell(lat_b, lon_b, res=9)
print(f"Hex A ID: {hex_id_a}")
print(f"Hex B ID: {hex_id_b}")
