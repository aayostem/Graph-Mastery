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


def min_knight_moves(board_size, start_pos, target_pos):
    """
    Finds the minimum number of moves a knight takes to reach a target on a chessboard.

    Args:
        board_size (int): Size of the square chessboard (e.g., 8 for 8x8).
        start_pos (tuple): (row, col) of the knight's starting position.
        target_pos (tuple): (row, col) of the target position.

    Returns:
        int: Minimum moves, or -1 if target is unreachable.
    """
    # Knight's possible moves (dr, dc)
    knight_moves = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ]

    # Check boundary
    def is_valid(r, c):
        return 0 <= r < board_size and 0 <= c < board_size

    queue = collections.deque([(start_pos[0], start_pos[1], 0)])  # (row, col, moves)
    visited = set()
    visited.add(start_pos)

    while queue:
        r, c, moves = queue.popleft()

        if (r, c) == target_pos:
            return moves

        for dr, dc in knight_moves:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, moves + 1))

    return -1  # Target unreachable
