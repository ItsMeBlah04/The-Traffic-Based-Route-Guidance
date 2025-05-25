import pandas as pd
import re
from haversine import haversine

# Read the CSV
df_lines = pd.read_csv("F:/Swinburne/COS30019/Assignment2B/datasets/graph_traffic.csv", quotechar='"', engine="python")
df_lines = df_lines[df_lines['WKT'].str.contains("LINESTRING", na=False)]

# Extract coordinate pairs from LINESTRING
def extract_coordinates(wkt):
    match = re.search(r"LINESTRING \(([^)]+)\)", wkt)
    if match:
        coords = match.group(1).split(", ")
        if len(coords) == 2:
            lon1, lat1 = map(float, coords[0].split())
            lon2, lat2 = map(float, coords[1].split())
            return (lat1, lon1), (lat2, lon2)
    return None, None

# Build nodes and edges
nodes_set = set()
edges = []

for wkt in df_lines['WKT']:
    coord1, coord2 = extract_coordinates(wkt)
    if coord1 and coord2:
        nodes_set.add(coord1)
        nodes_set.add(coord2)
        dist = round(haversine(coord1, coord2), 3)
        edges.append((coord1, coord2, dist))
        edges.append((coord2, coord1, dist))  # make it bidirectional

# Assign IDs
nodes_list = list(nodes_set)
node_id_map = {coord: i+1 for i, coord in enumerate(nodes_list)}

# Convert to ID-based edges with distance
edges_with_ids = [(node_id_map[a], node_id_map[b], d) for a, b, d in edges]

# Write map structure
lines = ["Nodes:"]
for coord, node_id in node_id_map.items():
    lat, lon = coord
    lines.append(f"{node_id}: ({lat},{lon})")

lines.append("\nEdges:")
for u, v, d in edges_with_ids:
    lines.append(f"({u},{v}): {d}")

# lines.append("\nOrigin:")
# lines.append(str(min(node_id_map.values())))

# lines.append("\nDestinations:")
# lines.append(str(max(node_id_map.values())))

# Save to file
with open("traffic_graph.txt", "w") as f:
    f.write("\n".join(lines))