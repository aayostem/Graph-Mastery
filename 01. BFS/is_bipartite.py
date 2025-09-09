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


def is_bipartite(num_vertices, adj_list):
    """
    Checks if an undirected graph is bipartite using BFS.

    Args:
        num_vertices (int): The total number of vertices.
        adj_list (dict): The adjacency list representation of the graph.

    Returns:
        bool: True if the graph is bipartite, False otherwise.
    """
    # 0: uncolored, 1: color A, -1: color B
    colors = [0] * num_vertices

    for i in range(num_vertices):
        if colors[i] == 0:  # If node is uncolored, start BFS from here
            queue = collections.deque([i])
            colors[i] = 1  # Assign first color

            while queue:
                u = queue.popleft()
                for v in adj_list[u]:
                    if colors[v] == 0:  # If neighbor is uncolored
                        colors[v] = -colors[u]  # Assign opposite color
                        queue.append(v)
                    elif (
                        colors[v] == colors[u]
                    ):  # If neighbor has same color, not bipartite
                        return False
    return True
