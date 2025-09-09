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


def walls_and_gates(rooms):
    """
    Fills empty rooms with the distance to the nearest gate using multi-source BFS.

    Args:
        rooms (list of lists): The 2D grid. -1: wall, 0: gate, 2147483647 (INF): empty room.
    """
    if not rooms or not rooms[0]:
        return

    rows, cols = len(rooms), len(rooms[0])
    queue = collections.deque()

    # Add all gates to the queue to start multi-source BFS
    for r in range(rows):
        for c in range(cols):
            if rooms[r][c] == 0:
                queue.append((r, c))  # Gates are at distance 0

    # Directions for 4-directional movement
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        r, c = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            # Check boundaries and if it's an empty room
            if 0 <= nr < rows and 0 <= nc < cols and rooms[nr][nc] == 2147483647:
                rooms[nr][nc] = rooms[r][c] + 1  # Update distance
                queue.append((nr, nc))  # Add to queue for further exploration
