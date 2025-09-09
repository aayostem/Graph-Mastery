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


def find_mother_vertex(num_vertices, edges):
    """
    Finds the mother vertex in a directed graph.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of lists): List of [u, v] directed edges.

    Returns:
        int: The label of the mother vertex, or -1 if none exists.
    """
    adj_list = collections.defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)

    last_finished_vertex = -1
    visited_dfs1 = [False] * num_vertices

    # Step 1: Perform DFS from all unvisited nodes to find the last finished vertex
    def dfs_util_1(u):
        visited_dfs1[u] = True
        for v in adj_list[u]:
            if not visited_dfs1[v]:
                dfs_util_1(v)
        nonlocal last_finished_vertex
        last_finished_vertex = u  # This vertex finishes last in its DFS component

    for i in range(num_vertices):
        if not visited_dfs1[i]:
            dfs_util_1(i)

    # Step 2: Perform DFS from the last_finished_vertex to check if all nodes are reachable
    if last_finished_vertex == -1:  # No vertices in graph
        return -1

    visited_dfs2 = [False] * num_vertices
    reachable_count = 0
    queue = collections.deque([last_finished_vertex])

    if 0 <= last_finished_vertex < num_vertices:
        visited_dfs2[last_finished_vertex] = True
        reachable_count += 1
    else:  # If last_finished_vertex is out of bounds (e.g., empty graph)
        return -1

    while queue:
        u = queue.popleft()
        for v in adj_list[u]:
            if not visited_dfs2[v]:
                visited_dfs2[v] = True
                reachable_count += 1
                queue.append(v)

    if reachable_count == num_vertices:
        return last_finished_vertex
    else:
        return -1
