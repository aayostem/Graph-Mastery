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


def has_cycle_directed(num_vertices, adj_list):
    """
    Checks if a directed graph has a cycle using DFS.

    Args:
        num_vertices (int): The total number of vertices.
        adj_list (dict): The adjacency list representation of the graph.

    Returns:
        bool: True if a cycle exists, False otherwise.
    """
    # 0: unvisited, 1: visiting (in current recursion stack), 2: visited (processed)
    visited_state = [0] * num_vertices

    def dfs_check_cycle_directed(u):
        visited_state[u] = 1  # Mark as visiting
        for v in adj_list[u]:
            if (
                visited_state[v] == 1
            ):  # Found a back-edge to a node in current recursion stack
                return True
            if visited_state[v] == 0:  # If unvisited, recurse
                if dfs_check_cycle_directed(v):
                    return True
        visited_state[u] = 2  # Mark as visited (processed)
        return False

    for i in range(num_vertices):
        if visited_state[i] == 0:
            if dfs_check_cycle_directed(i):
                return True
    return False
