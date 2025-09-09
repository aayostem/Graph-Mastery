import collections
import heapq


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def represent_graph(num_vertices, edges, is_directed=False, is_weighted=False):
    adj_list = collections.defaultdict(list)
    for edge in edges:
        u, v = edge[0], edge[1]
        weight = edge[2] if is_weighted else 1
        adj_list[u].append((v, weight))
        if not is_directed:
            adj_list[v].append((u, weight))
    return dict(adj_list)


def topological_sort(num_vertices, edges):
    """
    Performs a topological sort on a DAG using Kahn's algorithm (BFS-based).

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of tuples): List of (u, v) tuples representing directed edges.

    Returns:
        list: A list of nodes in topological order, or an empty list if a cycle is detected.
    """
    adj_list = represent_graph(num_vertices, edges, is_directed=True)
    in_degree = [0] * num_vertices

    for u in range(num_vertices):
        for v, _ in adj_list.get(u, []):
            in_degree[v] += 1

    queue = collections.deque()
    for i in range(num_vertices):
        if in_degree[i] == 0:
            queue.append(i)

    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)

        for v, _ in adj_list.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(top_order) == num_vertices:
        return top_order
    else:
        return []  # Cycle detected
