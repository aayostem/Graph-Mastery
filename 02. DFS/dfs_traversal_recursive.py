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


def dfs_traversal_recursive(num_vertices, adj_list, start_node):
    """
    Performs a Depth-First Search traversal on a graph using recursion.

    Args:
        num_vertices (int): The total number of vertices in the graph.
        adj_list (dict): The adjacency list representation of the graph.
        start_node (int): The starting node for the traversal.

    Returns:
        list: A list of nodes in DFS traversal order.
    """
    visited = [False] * num_vertices
    traversal_order = []

    def dfs_util(u):
        visited[u] = True
        traversal_order.append(u)
        for v in adj_list[u]:
            if not visited[v]:
                dfs_util(v)

    if 0 <= start_node < num_vertices:
        dfs_util(start_node)

    return traversal_order
