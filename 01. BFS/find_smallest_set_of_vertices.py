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


def find_smallest_set_of_vertices(n, edges):
    """
    Finds the smallest set of vertices from which all other nodes in a DAG are reachable.

    Args:
        n (int): The number of vertices.
        edges (list of lists): List of [u, v] directed edges.

    Returns:
        list: The smallest set of starting vertices.
    """
    # In a DAG, a node needs to be in the starting set only if no other node can reach it.
    # This means, nodes with an in-degree of 0 are the required starting points.

    in_degree = [0] * n
    for u, v in edges:
        in_degree[v] += 1

    result = []
    for i in range(n):
        if in_degree[i] == 0:
            result.append(i)

    return result
