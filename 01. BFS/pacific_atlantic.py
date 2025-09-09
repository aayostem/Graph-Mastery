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


def pacific_atlantic(heights):
    """
    Finds grid coordinates where water can flow to both Pacific and Atlantic oceans.

    Args:
        heights (list of lists): The 2D matrix of heights.

    Returns:
        list of lists: List of [row, col] coordinates.
    """
    if not heights or not heights[0]:
        return []

    rows, cols = len(heights), len(heights[0])

    # pacific_reachable[r][c] is True if water from (r,c) can reach Pacific
    # atlantic_reachable[r][c] is True if water from (r,c) can reach Atlantic
    pacific_reachable = [[False] * cols for _ in range(rows)]
    atlantic_reachable = [[False] * cols for _ in range(rows)]

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def bfs_flow(start_r, start_c, reachable_matrix):
        q = collections.deque([(start_r, start_c)])
        reachable_matrix[start_r][start_c] = True

        while q:
            r, c = q.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                # Check boundaries and if not yet visited
                # Crucial rule: water can flow from a cell to an adjacent one if the adjacent cell's height is less than or equal to the current cell's height.
                # When doing reverse BFS FROM ocean, water "flows" from lower/equal height to higher/equal height.
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and not reachable_matrix[nr][nc]
                    and heights[nr][nc] >= heights[r][c]
                ):  # Flow upwards/flat from ocean to higher land

                    reachable_matrix[nr][nc] = True
                    q.append((nr, nc))

    # Start BFS from Pacific border (top row, left column)
    for c in range(cols):
        bfs_flow(0, c, pacific_reachable)  # Top row
    for r in range(rows):
        bfs_flow(r, 0, pacific_reachable)  # Left column

    # Start BFS from Atlantic border (bottom row, right column)
    for c in range(cols):
        bfs_flow(rows - 1, c, atlantic_reachable)  # Bottom row
    for r in range(rows):
        bfs_flow(r, cols - 1, atlantic_reachable)  # Right column

    result = []
    for r in range(rows):
        for c in range(cols):
            if pacific_reachable[r][c] and atlantic_reachable[r][c]:
                result.append([r, c])

    return result
