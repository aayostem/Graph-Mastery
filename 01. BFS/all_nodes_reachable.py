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


def all_nodes_reachable(num_vertices, adj_list, start_node):
    """
    Checks if all nodes in a directed graph are reachable from a given start node.

    Args:
        num_vertices (int): The total number of vertices.
        adj_list (dict): The adjacency list representation of the graph.
        start_node (int): The starting node.

    Returns:
        bool: True if all nodes are reachable, False otherwise.
    """
    if not (0 <= start_node < num_vertices):
        return False  # Invalid start node

    visited_count = 0
    visited = [False] * num_vertices
    queue = collections.deque([start_node])

    visited[start_node] = True
    visited_count += 1

    while queue:
        u = queue.popleft()
        for v in adj_list[u]:
            if not visited[v]:
                visited[v] = True
                visited_count += 1
                queue.append(v)

    return visited_count == num_vertices
