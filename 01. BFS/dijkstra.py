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


def dijkstra(num_vertices, edges, source, destination):
    """
    Finds the shortest path in a weighted graph with non-negative edges using Dijkstra's algorithm.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of tuples): List of (u, v, weight) tuples representing weighted edges.
        source (int): The starting node.
        destination (int): The target node.

    Returns:
        int: The length of the shortest path, or float('inf') if no path exists.
    """
    adj_list = represent_graph(num_vertices, edges, is_directed=False, is_weighted=True)

    distances = {i: float("inf") for i in range(num_vertices)}
    distances[source] = 0

    # Priority queue stores tuples of (distance, node)
    priority_queue = [(0, source)]  # (current_distance, current_node)

    while priority_queue:
        current_distance, u = heapq.heappop(priority_queue)

        # If we have found a shorter path to u already, skip
        if current_distance > distances[u]:
            continue

        if u == destination:
            return distances[u]

        for v, weight in adj_list.get(u, []):
            distance = current_distance + weight
            # If a shorter path to v is found
            if distance < distances[v]:
                distances[v] = distance
                heapq.heappush(priority_queue, (distance, v))

    return distances[destination]  # Returns float('inf') if unreachable
