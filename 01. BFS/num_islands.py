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


def num_islands(grid):
    """
    Counts the number of islands in a 2D binary grid.

    Args:
        grid (list of lists): The 2D grid (list of strings or list of lists of chars).

    Returns:
        int: The number of islands.
    """
    if not grid or not grid[0]:
        return 0

    rows = len(grid)
    cols = len(grid[0])
    num_islands_count = 0

    # Define directions for 4-directional movement
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def bfs_sink_island(r, c):
        q = collections.deque([(r, c)])
        grid[r][c] = "0"  # Sink the current land piece

        while q:
            curr_r, curr_c = q.popleft()
            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "1":
                    grid[nr][nc] = "0"  # Sink
                    q.append((nr, nc))

    # Convert grid elements to mutable type (e.g., list of lists of chars) if they are strings
    mutable_grid = [list(row) for row in grid]

    for r in range(rows):
        for c in range(cols):
            if mutable_grid[r][c] == "1":
                num_islands_count += 1
                bfs_sink_island(
                    r, c
                )  # Use BFS to mark all connected land as visited ('0')

    return num_islands_count
