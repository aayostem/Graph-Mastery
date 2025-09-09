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


def is_valid_tree(n, edges):
    """
    Checks if a given graph (n nodes, n-1 edges) forms a valid tree.

    Args:
        n (int): The number of nodes.
        edges (list of lists): List of [u, v] edges.

    Returns:
        bool: True if it's a valid tree, False otherwise.
    """
    # A graph is a valid tree if and only if:
    # 1. It has exactly n-1 edges. (Given in problem statement, but good to double check)
    # 2. It is connected.
    # 3. It contains no cycles.

    if len(edges) != n - 1 and n != 1:  # For n=1, 0 edges means a valid tree
        return False
    if n == 1:
        return len(edges) == 0  # Single node, no edges, is a tree.

    adj_list = collections.defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    visited = [False] * n

    # DFS to check for cycles and connectivity
    def dfs_check(u, parent):
        visited[u] = True
        for v in adj_list[u]:
            if not visited[v]:
                if dfs_check(v, u):
                    return True  # Cycle detected in subtree
            elif v != parent:
                return True  # Back-edge to a non-parent, thus a cycle
        return False

    # Start DFS from node 0 (assuming node 0 exists and is part of the graph)
    # If the graph has isolated nodes not connected to node 0, this will detect it as not connected.
    if dfs_check(0, -1):  # Check for cycle starting from 0
        return False

    # After DFS, check if all nodes were visited (connectivity)
    for i in range(n):
        if not visited[i]:
            return False  # Not all nodes are connected

    return True  # Passed all checks
