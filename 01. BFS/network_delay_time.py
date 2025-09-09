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


def network_delay_time(times, n, k):
    """
    Calculates the time it takes for all nodes to receive a signal from node k.

    Args:
        times (list of lists): List of [u, v, w] where u->v with weight w.
        n (int): Total number of nodes (labeled 1 to n).
        k (int): Starting node for the signal (labeled 1 to n).

    Returns:
        int: The maximum time for all nodes to receive the signal, or -1 if some nodes are unreachable.
    """
    # Build adjacency list (1-indexed nodes, convert to 0-indexed for array access internally)
    adj_list = collections.defaultdict(list)
    for u, v, w in times:
        adj_list[u - 1].append((v - 1, w))  # Convert to 0-indexed

    distances = {i: float("inf") for i in range(n)}
    distances[k - 1] = 0  # Adjust k to be 0-indexed

    # Priority queue: (current_distance, current_node)
    priority_queue = [(0, k - 1)]

    while priority_queue:
        current_distance, u = heapq.heappop(priority_queue)

        if current_distance > distances[u]:
            continue

        for v, weight in adj_list.get(u, []):
            distance = current_distance + weight
            if distance < distances[v]:
                distances[v] = distance
                heapq.heappush(priority_queue, (distance, v))

    max_time = 0
    for i in range(n):
        if distances[i] == float("inf"):
            return -1  # Node is unreachable
        max_time = max(max_time, distances[i])

    return max_time
