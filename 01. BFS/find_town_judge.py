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


def find_town_judge(n, trust):
    """
    Finds the town judge.

    Args:
        n (int): The number of people in the town (labeled from 1 to n).
        trust (list of lists): trust[i] = [a, b] representing that person 'a' trusts person 'b'.

    Returns:
        int: The label of the town judge, or -1 if no such person exists.
    """
    # Adjust for 0-indexed if necessary, but problem states 1 to N, so use N+1 size arrays
    in_degree = [0] * (n + 1)
    out_degree = [0] * (n + 1)

    for a, b in trust:
        out_degree[a] += 1
        in_degree[b] += 1

    for i in range(1, n + 1):  # Iterate from 1 to n
        if in_degree[i] == n - 1 and out_degree[i] == 0:
            return i

    return -1
