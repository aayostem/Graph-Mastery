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


def find_star_center(edges):
    """
    Finds the center of a star graph.

    Args:
        edges (list of tuples): A list of (u, v) tuples representing edges.

    Returns:
        int: The central node of the star graph, or -1 if not a valid star graph (simplified).
    """
    # In a star graph, the center node must be present in every edge.
    # So, we just need to find the common node in the first two edges.
    if not edges or len(edges) < 2:
        # A star graph with N vertices has N-1 edges. Minimum 2 edges for unique center.
        return -1

    node1_edge1, node2_edge1 = edges[0]
    node1_edge2, node2_edge2 = edges[1]

    # The common node must be the center
    if node1_edge1 == node1_edge2 or node1_edge1 == node2_edge2:
        return node1_edge1
    if node2_edge1 == node1_edge2 or node2_edge1 == node2_edge2:
        return node2_edge1

    return -1  # Should not happen for a valid star graph
