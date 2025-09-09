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


def min_reorder(n, connections):
    """
    Calculates the minimum number of road reorientations needed for all paths to lead to city 0.

    Args:
        n (int): The number of cities (nodes).
        connections (list of lists): List of [a, b] directed roads from a to b.

    Returns:
        int: The minimum number of reorientations.
    """
    # Create two adjacency lists: one for original directions, one for reversed (for traversal from 0)
    # adj[u] will store (v, is_original_direction)
    adj = collections.defaultdict(list)
    for u, v in connections:
        adj[u].append((v, 1))  # 1 indicates original direction (u -> v)
        adj[v].append((u, 0))  # 0 indicates reversed direction (v -> u)

    reorientations = 0
    visited = [False] * n

    # BFS to traverse from city 0
    queue = collections.deque([0])
    visited[0] = True

    while queue:
        u = queue.popleft()
        for v, is_original in adj[u]:
            if not visited[v]:
                visited[v] = True
                if (
                    is_original == 1
                ):  # If the original edge was u -> v, it points away from 0
                    reorientations += 1  # So, we need to reorient it
                queue.append(v)

    return reorientations
