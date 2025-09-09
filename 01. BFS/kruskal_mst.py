import collections
import heapq


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size  # For union by rank optimization

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by rank
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True  # Successfully united
        return False  # Already in the same set


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


def kruskal_mst(num_vertices, edges):
    """
    Finds the Minimum Spanning Tree (MST) of a connected, undirected graph using Kruskal's algorithm.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of lists): List of [u, v, weight] tuples representing weighted edges.

    Returns:
        tuple: (int, list) - The total weight of the MST and a list of edges in the MST.
                             Returns (0, []) if num_vertices is 0 or 1.
                             Returns (float('inf'), []) if graph is not connected (and MST not formed).
    """
    if num_vertices <= 1:
        return 0, []

    # Sort edges by weight
    sorted_edges = sorted(edges, key=lambda x: x[2])

    uf = UnionFind(num_vertices)
    mst_weight = 0
    mst_edges = []
    edges_in_mst_count = 0

    for u, v, weight in sorted_edges:
        if uf.union(u, v):
            mst_weight += weight
            mst_edges.append((u, v, weight))
            edges_in_mst_count += 1
            if (
                edges_in_mst_count == num_vertices - 1
            ):  # A tree with N vertices has N-1 edges
                break

    # Check if all vertices are connected (if graph was disconnected, we won't form N-1 edges)
    # This check is implicitly handled if you return the MST only when edges_in_mst_count == num_vertices - 1
    # If the original graph is guaranteed connected, this check is not strictly needed.
    # Otherwise, if it doesn't form (num_vertices - 1) edges, it's not fully connected.
    if edges_in_mst_count != num_vertices - 1 and num_vertices > 1:
        return (
            float("inf"),
            [],
        )  # Graph not connected, or not enough edges to form a spanning tree

    return mst_weight, mst_edges
