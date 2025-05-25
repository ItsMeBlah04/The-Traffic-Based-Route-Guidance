from pathfinders.pathfinders import PathFinders

origin = 3
goals = {52}

path_finder = PathFinders("F:/Swinburne/COS30019/Assignment2B/datasets/graph_map/traffic_graph.txt")
goal, created, path = path_finder.uniform_cost_search(origin, goals)
print(path)
print("Path:", "->".join(map(str, path)))
print("Cost:", path_finder.calculate_path_cost(path))