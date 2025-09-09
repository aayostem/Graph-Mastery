import collections
import heapq
from bfs_distances import *


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


def find_farthest_node_in_tree(num_vertices, edges, start_node_initial_bfs=0):
    """
    Finds the node farthest from a given starting node in a tree using two BFS traversals.
    Also finds the diameter of the tree.

    Args:
        num_vertices (int): The total number of vertices in the tree.
        edges (list of tuples): Edges of the tree.
        start_node_initial_bfs (int): An arbitrary starting node for the first BFS (e.g., 0).

    Returns:
        tuple: (farthest_node, max_distance) from the initial start_node.
    """
    adj_list, _ = represent_graph(num_vertices, edges, is_directed=False)

    # 1. BFS from an arbitrary node (e.g., node 0)
    distances_from_start = bfs_distances(num_vertices, adj_list, start_node_initial_bfs)

    # Find the node farthest from the initial start_node
    node_u = -1
    max_dist_u = -1
    for i in range(num_vertices):
        if distances_from_start[i] > max_dist_u:
            max_dist_u = distances_from_start[i]
            node_u = i

    # 2. BFS from node_u (which is one endpoint of the diameter)
    distances_from_u = bfs_distances(num_vertices, adj_list, node_u)

    # Find the node farthest from node_u (which is the other endpoint of the diameter)
    node_v = -1
    max_dist_v = -1
    for i in range(num_vertices):
        if distances_from_u[i] > max_dist_v:
            max_dist_v = distances_from_u[i]
            node_v = i

    # node_v is the farthest node from node_u (and thus from any node, it's an endpoint of diameter)
    # max_dist_v is the diameter

    # To answer the specific question "farthest from given starting node":
    # The farthest node from the initial_start_node is node_u, with distance max_dist_u.
    # The problem asks for the farthest node from *a given* starting node.
    # If the question meant "farthest node *in the tree overall*", then max_dist_v is the diameter.

    # Let's return node_u and max_dist_u, as it's the node farthest from the *initial* start_node_initial_bfs.
    return node_u, max_dist_u
