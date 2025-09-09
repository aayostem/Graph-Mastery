import collections
import heapq


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def represent_graph(num_vertices, edges, is_directed=False):
    """
    Converts a list of edges into an adjacency list and an adjacency matrix.

    Args:
        num_vertices (int): The total number of vertices in the graph.
        edges (list of tuples): A list of (u, v) tuples representing edges.
        is_directed (bool): True if the graph is directed, False otherwise.

    Returns:
        tuple: A tuple containing:
            - dict: Adjacency list representation.
            - list of lists: Adjacency matrix representation.
    """
    # Adjacency List
    adj_list = collections.defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
        if not is_directed:
            adj_list[v].append(u)

    # Adjacency Matrix
    adj_matrix = [[0] * num_vertices for _ in range(num_vertices)]
    for u, v in edges:
        adj_matrix[u][v] = 1
        if not is_directed:
            adj_matrix[v][u] = 1

    return dict(adj_list), adj_matrix


def find_cheapest_flight(n, flights, src, dst, k):
    """
    Finds the cheapest flight within K stops using a modified BFS (Bellman-Ford-like).

    Args:
        n (int): Number of cities.
        flights (list of lists): List of [u, v, price] flights.
        src (int): Source city.
        dst (int): Destination city.
        k (int): Maximum number of stops allowed.

    Returns:
        int: The cheapest price, or -1 if no such route exists.
    """
    adj_list = collections.defaultdict(list)
    for u, v, price in flights:
        adj_list[u].append((v, price))

    # distances[i] stores the minimum cost to reach city i
    distances = [float("inf")] * n
    distances[src] = 0

    # queue stores (cost, city, stops_taken)
    queue = collections.deque(
        [(0, src, 0)]
    )  # (price_so_far, current_city, stops_taken)

    min_price = float("inf")

    # To optimize and avoid redundant paths, especially for Bellman-Ford like behavior
    # where paths with more stops might become cheaper, we need a way to track
    # minimum cost for a given number of stops.
    # A distances array `min_cost_with_stops[stops_taken][city]` is better for explicit Bellman-Ford
    # For BFS-based, it's a bit trickier, need to make sure we don't relax using
    # a path that's already worse given the stops.

    # Let's use a standard BFS but with a visited state that includes stops
    # min_cost_reached[city][stops_taken] stores min cost to reach city with exact stops_taken
    min_cost_reached = [
        [float("inf")] * (k + 2) for _ in range(n)
    ]  # k+1 for max stops, +1 for 0-stops
    min_cost_reached[src][0] = 0

    while queue:
        current_price, u, stops_taken = queue.popleft()

        # If we exceeded max stops, or this path is already worse than known, skip
        if stops_taken > k + 1:  # We allow K stops, meaning K+1 edges
            continue

        # If we are already at a cheaper price for 'u' with 'stops_taken' stops, continue
        # This check is crucial for correctness with costs.
        if current_price > min_cost_reached[u][stops_taken]:
            continue

        if u == dst:
            min_price = min(min_price, current_price)
            continue  # Found a path, continue searching for potentially cheaper paths

        # Explore neighbors
        for v, price_to_v in adj_list.get(u, []):
            new_price = current_price + price_to_v
            new_stops_taken = stops_taken + 1

            # Only consider if within K stops and if this path offers a cheaper cost
            if (
                new_stops_taken <= k + 1
                and new_price < min_cost_reached[v][new_stops_taken]
            ):
                min_cost_reached[v][new_stops_taken] = new_price
                queue.append((new_price, v, new_stops_taken))

    return min_price if min_price != float("inf") else -1
