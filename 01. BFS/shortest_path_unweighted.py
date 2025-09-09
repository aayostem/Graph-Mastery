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


def shortest_path_unweighted(num_vertices, edges, source, destination):
    """
    Finds the shortest path (number of edges) between two nodes in an unweighted graph using BFS.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of tuples): List of (u, v) tuples representing edges.
        source (int): The starting node.
        destination (int): The target node.

    Returns:
        int: The length of the shortest path, or -1 if no path exists.
    """
    adj_list = represent_graph(
        num_vertices, edges, is_directed=False, is_weighted=False
    )

    if source == destination:
        return 0

    distances = [-1] * num_vertices
    queue = collections.deque()

    if 0 <= source < num_vertices:
        queue.append(source)
        distances[source] = 0
    else:
        return -1  # Invalid source

    while queue:
        u = queue.popleft()

        # In adj_list, if not weighted, we store (neighbor, 1) or just neighbor
        # Let's assume edges are stored as (v, weight) and weight is 1 for unweighted
        for v, weight in adj_list.get(u, []):
            if distances[v] == -1:  # If not visited
                distances[v] = distances[u] + 1
                queue.append(v)
                if v == destination:
                    return distances[v]

    return -1  # Destination unreachable
