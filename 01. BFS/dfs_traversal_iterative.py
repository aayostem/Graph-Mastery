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


def dfs_traversal_iterative(num_vertices, adj_list, start_node):
    """
    Performs a Depth-First Search traversal on a graph using an iterative approach (stack).

    Args:
        num_vertices (int): The total number of vertices in the graph.
        adj_list (dict): The adjacency list representation of the graph.
        start_node (int): The starting node for the traversal.

    Returns:
        list: A list of nodes in DFS traversal order.
    """
    visited = [False] * num_vertices
    stack = [start_node]
    traversal_order = []

    if not (0 <= start_node < num_vertices):
        return []

    visited[start_node] = True
    traversal_order.append(start_node)

    while stack:
        u = stack[-1]  # Peek at the top of the stack

        found_unvisited_neighbor = False
        for v in adj_list[u]:
            if not visited[v]:
                visited[v] = True
                traversal_order.append(v)
                stack.append(v)
                found_unvisited_neighbor = True
                break  # Explore this neighbor before popping current node

        if not found_unvisited_neighbor:
            stack.pop()  # Backtrack if all neighbors are visited or no neighbors

    return traversal_order
