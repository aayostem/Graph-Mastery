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


def has_cycle_undirected(num_vertices, adj_list):
    """
    Checks if an undirected graph has a cycle using DFS.

    Args:
        num_vertices (int): The total number of vertices.
        adj_list (dict): The adjacency list representation of the graph.

    Returns:
        bool: True if a cycle exists, False otherwise.
    """
    visited = [False] * num_vertices

    def dfs_check_cycle(u, parent):
        visited[u] = True
        for v in adj_list[u]:
            if not visited[v]:
                if dfs_check_cycle(v, u):  # Pass current node as parent for neighbor
                    return True
            elif (
                v != parent
            ):  # If visited and not parent, it's a back-edge to an ancestor, hence a cycle
                return True
        return False

    for i in range(num_vertices):
        if not visited[i]:
            if dfs_check_cycle(i, -1):  # -1 indicates no parent for the initial call
                return True
    return False
