from route_planner import RoutePlanner

# Instantiate the planner
route_planner = RoutePlanner()

# Define inputs
origin = "kilby_rd"
destination = "burnley_st"
date = "05/10/2006"        # Format: dd/mm/yyyy
time = "09:00"             # Must be at 15-minute interval (e.g., 08:00, 08:15)
model_type = "cnn_lstm"         # Options: 'gru', 'lstm', 'cnn_lstm'
path_finder_type = "ucs"   # Options: 'bfs', 'dfs', 'ucs', 'a_star', etc.

# try:
# Run route estimation
locations, coordinates, est_time, cost, total_flow, speeds = route_planner.route_estimate(
    origin=origin,
    destination=destination,
    date=date,
    time=time,
    model_type=model_type,
    path_finder_type=path_finder_type
)

# Display results
print("\n🗺️  Route Details")
print("----------------------")
print("Route Locations:")
for loc in locations:
    print(f"  - {loc}")
print("\nCoordinates:")
for lat, lon in coordinates:
    print(f"  - ({lat:.6f}, {lon:.6f})")

print(f"\n🚦 Total Flow: {total_flow:.2f} vehicles/h")
print(f"🚗 Estimated Speeds [congested, free-flow]: {speeds[0]:.2f} km/h, {speeds[1]:.2f} km/h")
print(f"🕒 Estimated Travel Time: {est_time:.2f} minutes")
print(f"📏 Path Cost (Distance): {cost:.2f} km")

# except ValueError as e:
#     print(f"❌ Error: {e}")
# except FileNotFoundError as e:
#     print(f"❌ Missing file: {e.filename}")