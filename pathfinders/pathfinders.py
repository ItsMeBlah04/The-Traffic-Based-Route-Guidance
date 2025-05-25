from collections import deque, defaultdict
from heapq import heappush, heappop
from math import sqrt

class PathFinders:
    def __init__(self, filename):
        self.graph = defaultdict(list)
        self.nodes = {}  
        self.parse_input_file(filename)

    def parse_input_file(self, filename):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        section = None
        for line in lines:
            if line.startswith("Nodes:"):
                section = "nodes"
            elif line.startswith("Edges:"):
                section = "edges"
            elif section == "nodes":
                node_id, coord = line.split(":")
                node_id = int(node_id.strip())
                self.nodes[node_id] = tuple(map(float, coord.strip().strip("()").split(",")))
            elif section == "edges":
                edge_part, cost = line.split(":")
                n1, n2 = map(int, edge_part.strip("()").split(","))
                self.graph[n1].append((n2, float(cost.strip())))

    def bfs(self, origin, goals):
        visited, queue = set(), deque([(origin, [origin])])
        visited.add(origin)
        nodes_created = 1

        while queue:
            current, path = queue.popleft()
            if current in goals:
                return current, nodes_created, path
            for neighbor, _ in sorted(self.graph[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    nodes_created += 1
        return None, nodes_created, []

    def dfs(self, origin, goals):
        visited, stack = set(), [(origin, [origin])]
        visited.add(origin)
        nodes_created = 1

        while stack:
            current, path = stack.pop()
            if current in goals:
                return current, nodes_created, path
            for neighbor, _ in sorted(self.graph[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))
                    nodes_created += 1
        return None, nodes_created, []

    def heuristic(self, node, goals):
        x1, y1 = self.nodes[node]
        return min(sqrt((x1 - x2)**2 + (y1 - y2)**2) for (x2, y2) in [self.nodes[g] for g in goals])

    def gbfs(self, origin, goals):
        frontier = [(origin, self.heuristic(origin, goals), [origin])]
        visited = set()
        nodes_created = 1

        while frontier:
            frontier.sort(key=lambda x: (x[1], x[0]))
            current, _, path = frontier.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current in goals:
                return current, nodes_created, path
            for neighbor, _ in sorted(self.graph.get(current, [])):
                if neighbor not in visited:
                    h = self.heuristic(neighbor, goals)
                    frontier.append((neighbor, h, path + [neighbor]))
                    nodes_created += 1
        return None, nodes_created, []

    def astar(self, origin, goals):
        h_start = self.heuristic(origin, goals)
        pq = [(h_start, 0, origin, [origin])]
        visited = set()
        nodes_created = 1

        while pq:
            f, g, node, path = heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            if node in goals:
                return node, nodes_created, path
            for neighbor, cost in self.graph[node]:
                if neighbor not in visited:
                    g_new = g + cost
                    h = self.heuristic(neighbor, goals)
                    heappush(pq, (g_new + h, g_new, neighbor, path + [neighbor]))
                    nodes_created += 1
        return None, nodes_created, []

    def df_limited(self, origin, goals, limit=1000):
        visited, stack = set(), [(origin, [origin], 0)]
        visited.add(origin)
        nodes_created = 1

        while stack:
            current, path, depth = stack.pop()
            if current in goals:
                return current, nodes_created, path
            if depth < limit:
                for neighbor, _ in sorted(self.graph[current]):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append((neighbor, path + [neighbor], depth + 1))
                        nodes_created += 1
        return None, nodes_created, []

    def uniform_cost_search(self, origin, goals):
        costs = {node: float('inf') for node in self.nodes}
        costs[origin] = 0
        previous = {node: None for node in self.nodes}
        visited = set()
        pq = [(0, origin)]
        nodes_created = 0

        while pq:
            current_cost, current = heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            nodes_created += 1
            if current in goals:
                path = []
                while current:
                    path.append(current)
                    current = previous[current]
                return path[-1], nodes_created, path[::-1]
            for neighbor, weight in self.graph[current]:
                if neighbor in visited:
                    continue
                new_cost = current_cost + weight
                if new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    previous[neighbor] = current
                    heappush(pq, (new_cost, neighbor))
        return None, nodes_created, []

    def calculate_path_cost(self, path):
        total = 0
        for i in range(len(path) - 1):
            for neighbor, cost in self.graph[path[i]]:
                if neighbor == path[i + 1]:
                    total += cost
                    break
        return total