import pandas as pd

# Load your .txt file (assumes standard structure)
def parse_nodes_from_txt(file_path):
    nodes = {}
    with open(file_path, "r") as f:
        lines = f.readlines()

    reading_nodes = False
    for line in lines:
        if line.strip().upper().startswith("NODES:"):
            reading_nodes = True
            continue
        if line.strip().upper().startswith("EDGES:"):
            break
        if reading_nodes and ":" in line:
            node_id, coord = line.split(":")
            lat, lon = map(float, coord.strip().strip("()").split(","))
            nodes[int(node_id)] = (round(lat, 6), round(lon, 6))
    return nodes

# Load your CSV of known coordinates
def match_nodes_to_locations(txt_nodes, csv_path):
    df_csv = pd.read_csv(csv_path)
    df_csv['Latitude'] = df_csv['Latitude'].round(6)
    df_csv['Longitude'] = df_csv['Longitude'].round(6)
    
    coord_to_name = {
        (row['Latitude'], row['Longitude']): row['Location']
        for _, row in df_csv.iterrows()
    }

    result = []
    for node_id, coord in txt_nodes.items():
        name = coord_to_name.get(coord, "UNKNOWN")
        result.append((node_id, coord[0], coord[1], name))
    
    return pd.DataFrame(result, columns=["Node ID", "Latitude", "Longitude", "Location"])

# Usage
txt_nodes = parse_nodes_from_txt("datasets/graph_map/traffic_graph.txt")
matched_df = match_nodes_to_locations(txt_nodes, "datasets/graph_locations.csv")
matched_df.to_csv("node_id_to_location.csv", index=False)