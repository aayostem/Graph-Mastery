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


def bfs_traversal(num_vertices, adj_list, start_node):
    """
    Performs a Breadth-First Search traversal on a graph.

    Args:
        num_vertices (int): The total number of vertices in the graph.
        adj_list (dict): The adjacency list representation of the graph.
        start_node (int): The starting node for the traversal.

    Returns:
        list: A list of nodes in BFS traversal order.
    """
    visited = [False] * num_vertices
    queue = collections.deque()
    traversal_order = []

    if 0 <= start_node < num_vertices:
        queue.append(start_node)
        visited[start_node] = True
        traversal_order.append(start_node)

    while queue:
        u = queue.popleft()
        for v in adj_list[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
                traversal_order.append(v)

    return traversal_order


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


def path_exists(num_vertices, adj_list, source, destination):
    """
    Checks if a path exists between two nodes using BFS.

    Args:
        num_vertices (int): The total number of vertices.
        adj_list (dict): The adjacency list of the graph.
        source (int): The starting node.
        destination (int): The target node.

    Returns:
        bool: True if a path exists, False otherwise.
    """
    if source == destination:
        return True

    visited = [False] * num_vertices
    queue = collections.deque()

    if 0 <= source < num_vertices:
        queue.append(source)
        visited[source] = True

    while queue:
        u = queue.popleft()
        for v in adj_list[u]:
            if v == destination:
                return True
            if not visited[v]:
                visited[v] = True
                queue.append(v)

    return False


def count_connected_components(num_vertices, adj_list):
    """
    Counts the number of connected components in an undirected graph.

    Args:
        num_vertices (int): The total number of vertices.
        adj_list (dict): The adjacency list representation of the graph.

    Returns:
        int: The number of connected components.
    """
    visited = [False] * num_vertices
    components = 0

    def bfs_component(start_node):
        queue = collections.deque([start_node])
        visited[start_node] = True
        while queue:
            u = queue.popleft()
            for v in adj_list[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

    for i in range(num_vertices):
        if not visited[i]:
            components += 1
            bfs_component(i)  # Can use DFS here too

    return components


def find_star_center(edges):
    """
    Finds the center of a star graph.

    Args:
        edges (list of tuples): A list of (u, v) tuples representing edges.

    Returns:
        int: The central node of the star graph, or -1 if not a valid star graph (simplified).
    """
    # In a star graph, the center node must be present in every edge.
    # So, we just need to find the common node in the first two edges.
    if not edges or len(edges) < 2:
        # A star graph with N vertices has N-1 edges. Minimum 2 edges for unique center.
        return -1

    node1_edge1, node2_edge1 = edges[0]
    node1_edge2, node2_edge2 = edges[1]

    # The common node must be the center
    if node1_edge1 == node1_edge2 or node1_edge1 == node2_edge2:
        return node1_edge1
    if node2_edge1 == node1_edge2 or node2_edge1 == node2_edge2:
        return node2_edge1

    return -1  # Should not happen for a valid star graph


def is_bipartite(num_vertices, adj_list):
    """
    Checks if an undirected graph is bipartite using BFS.

    Args:
        num_vertices (int): The total number of vertices.
        adj_list (dict): The adjacency list representation of the graph.

    Returns:
        bool: True if the graph is bipartite, False otherwise.
    """
    # 0: uncolored, 1: color A, -1: color B
    colors = [0] * num_vertices

    for i in range(num_vertices):
        if colors[i] == 0:  # If node is uncolored, start BFS from here
            queue = collections.deque([i])
            colors[i] = 1  # Assign first color

            while queue:
                u = queue.popleft()
                for v in adj_list[u]:
                    if colors[v] == 0:  # If neighbor is uncolored
                        colors[v] = -colors[u]  # Assign opposite color
                        queue.append(v)
                    elif (
                        colors[v] == colors[u]
                    ):  # If neighbor has same color, not bipartite
                        return False
    return True


def find_town_judge(n, trust):
    """
    Finds the town judge.

    Args:
        n (int): The number of people in the town (labeled from 1 to n).
        trust (list of lists): trust[i] = [a, b] representing that person 'a' trusts person 'b'.

    Returns:
        int: The label of the town judge, or -1 if no such person exists.
    """
    # Adjust for 0-indexed if necessary, but problem states 1 to N, so use N+1 size arrays
    in_degree = [0] * (n + 1)
    out_degree = [0] * (n + 1)

    for a, b in trust:
        out_degree[a] += 1
        in_degree[b] += 1

    for i in range(1, n + 1):  # Iterate from 1 to n
        if in_degree[i] == n - 1 and out_degree[i] == 0:
            return i

    return -1


def all_nodes_reachable(num_vertices, adj_list, start_node):
    """
    Checks if all nodes in a directed graph are reachable from a given start node.

    Args:
        num_vertices (int): The total number of vertices.
        adj_list (dict): The adjacency list representation of the graph.
        start_node (int): The starting node.

    Returns:
        bool: True if all nodes are reachable, False otherwise.
    """
    if not (0 <= start_node < num_vertices):
        return False  # Invalid start node

    visited_count = 0
    visited = [False] * num_vertices
    queue = collections.deque([start_node])

    visited[start_node] = True
    visited_count += 1

    while queue:
        u = queue.popleft()
        for v in adj_list[u]:
            if not visited[v]:
                visited[v] = True
                visited_count += 1
                queue.append(v)

    return visited_count == num_vertices


def bfs_distances(num_vertices, adj_list, start_node):
    """Helper function to run BFS and return distances from start_node."""
    distances = [-1] * num_vertices
    queue = collections.deque([(start_node, 0)])  # (node, distance)
    distances[start_node] = 0

    while queue:
        u, d = queue.popleft()
        for v in adj_list[u]:
            if distances[v] == -1:  # If not visited
                distances[v] = d + 1
                queue.append((v, d + 1))
    return distances


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


def is_valid_tree(n, edges):
    """
    Checks if a given graph (n nodes, n-1 edges) forms a valid tree.

    Args:
        n (int): The number of nodes.
        edges (list of lists): List of [u, v] edges.

    Returns:
        bool: True if it's a valid tree, False otherwise.
    """
    # A graph is a valid tree if and only if:
    # 1. It has exactly n-1 edges. (Given in problem statement, but good to double check)
    # 2. It is connected.
    # 3. It contains no cycles.

    if len(edges) != n - 1 and n != 1:  # For n=1, 0 edges means a valid tree
        return False
    if n == 1:
        return len(edges) == 0  # Single node, no edges, is a tree.

    adj_list = collections.defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    visited = [False] * n

    # DFS to check for cycles and connectivity
    def dfs_check(u, parent):
        visited[u] = True
        for v in adj_list[u]:
            if not visited[v]:
                if dfs_check(v, u):
                    return True  # Cycle detected in subtree
            elif v != parent:
                return True  # Back-edge to a non-parent, thus a cycle
        return False

    # Start DFS from node 0 (assuming node 0 exists and is part of the graph)
    # If the graph has isolated nodes not connected to node 0, this will detect it as not connected.
    if dfs_check(0, -1):  # Check for cycle starting from 0
        return False

    # After DFS, check if all nodes were visited (connectivity)
    for i in range(n):
        if not visited[i]:
            return False  # Not all nodes are connected

    return True  # Passed all checks


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


def dijkstra(num_vertices, edges, source, destination):
    """
    Finds the shortest path in a weighted graph with non-negative edges using Dijkstra's algorithm.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of tuples): List of (u, v, weight) tuples representing weighted edges.
        source (int): The starting node.
        destination (int): The target node.

    Returns:
        int: The length of the shortest path, or float('inf') if no path exists.
    """
    adj_list = represent_graph(num_vertices, edges, is_directed=False, is_weighted=True)

    distances = {i: float("inf") for i in range(num_vertices)}
    distances[source] = 0

    # Priority queue stores tuples of (distance, node)
    priority_queue = [(0, source)]  # (current_distance, current_node)

    while priority_queue:
        current_distance, u = heapq.heappop(priority_queue)

        # If we have found a shorter path to u already, skip
        if current_distance > distances[u]:
            continue

        if u == destination:
            return distances[u]

        for v, weight in adj_list.get(u, []):
            distance = current_distance + weight
            # If a shorter path to v is found
            if distance < distances[v]:
                distances[v] = distance
                heapq.heappush(priority_queue, (distance, v))

    return distances[destination]  # Returns float('inf') if unreachable


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


def can_finish_courses(num_courses, prerequisites):
    """
    Determines if it's possible to take all courses given prerequisites (cycle detection).

    Args:
        num_courses (int): Total number of courses.
        prerequisites (list of lists): List of [course, prerequisite] pairs.

    Returns:
        bool: True if possible to take all courses, False otherwise.
    """
    # This is equivalent to checking if the graph formed by courses and prerequisites is a DAG.
    # We can use Kahn's algorithm or DFS-based cycle detection.
    # Let's use Kahn's algorithm.

    adj_list = collections.defaultdict(list)
    in_degree = [0] * num_courses

    for course, pre_req in prerequisites:
        adj_list[pre_req].append((course, 1))  # Edge from pre_req to course
        in_degree[course] += 1

    queue = collections.deque()
    for i in range(num_courses):
        if in_degree[i] == 0:
            queue.append(i)

    courses_taken = 0
    while queue:
        u = queue.popleft()
        courses_taken += 1

        for v, _ in adj_list.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return courses_taken == num_courses


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


def clone_graph(node: "Node") -> "Node":
    """
    Clones an undirected graph (deep copy).

    Args:
        node (Node): The reference to a node in the original graph.

    Returns:
        Node: The reference to the corresponding node in the cloned graph.
    """
    if not node:
        return None

    # Use a dictionary to store mapping from original node to cloned node
    # to handle cycles and avoid re-cloning already cloned nodes.
    cloned_nodes = {}  # original_node.val -> cloned_node

    # BFS approach
    queue = collections.deque([node])
    cloned_nodes[node.val] = Node(node.val)  # Clone the starting node

    while queue:
        curr_original = queue.popleft()
        curr_cloned = cloned_nodes[curr_original.val]

        for neighbor_original in curr_original.neighbors:
            if neighbor_original.val not in cloned_nodes:
                # If neighbor not cloned yet, clone it and add to queue
                cloned_nodes[neighbor_original.val] = Node(neighbor_original.val)
                queue.append(neighbor_original)

            # Add the cloned neighbor to the current cloned node's neighbors list
            curr_cloned.neighbors.append(cloned_nodes[neighbor_original.val])

    return cloned_nodes[node.val]


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


def network_delay_time(times, n, k):
    """
    Calculates the time it takes for all nodes to receive a signal from node k.

    Args:
        times (list of lists): List of [u, v, w] where u->v with weight w.
        n (int): Total number of nodes (labeled 1 to n).
        k (int): Starting node for the signal (labeled 1 to n).

    Returns:
        int: The maximum time for all nodes to receive the signal, or -1 if some nodes are unreachable.
    """
    # Build adjacency list (1-indexed nodes, convert to 0-indexed for array access internally)
    adj_list = collections.defaultdict(list)
    for u, v, w in times:
        adj_list[u - 1].append((v - 1, w))  # Convert to 0-indexed

    distances = {i: float("inf") for i in range(n)}
    distances[k - 1] = 0  # Adjust k to be 0-indexed

    # Priority queue: (current_distance, current_node)
    priority_queue = [(0, k - 1)]

    while priority_queue:
        current_distance, u = heapq.heappop(priority_queue)

        if current_distance > distances[u]:
            continue

        for v, weight in adj_list.get(u, []):
            distance = current_distance + weight
            if distance < distances[v]:
                distances[v] = distance
                heapq.heappush(priority_queue, (distance, v))

    max_time = 0
    for i in range(n):
        if distances[i] == float("inf"):
            return -1  # Node is unreachable
        max_time = max(max_time, distances[i])

    return max_time


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


def find_cheapest_flight(n, flights, src, dst, k):
    """
    Finds the cheapest flight within K stops using a modified BFS (Bellman-Ford-like).

    Args:
        n (int): Number of cities.
        flights (list of lists): List of [u, v, price] flights.
        src (int): Source city.
        dst (int): Destination city.
        k (int): Maximum number of stops allowed.

    Returns:
        int: The cheapest price, or -1 if no such route exists.
    """
    adj_list = collections.defaultdict(list)
    for u, v, price in flights:
        adj_list[u].append((v, price))

    # distances[i] stores the minimum cost to reach city i
    distances = [float("inf")] * n
    distances[src] = 0

    # queue stores (cost, city, stops_taken)
    queue = collections.deque(
        [(0, src, 0)]
    )  # (price_so_far, current_city, stops_taken)

    min_price = float("inf")

    # To optimize and avoid redundant paths, especially for Bellman-Ford like behavior
    # where paths with more stops might become cheaper, we need a way to track
    # minimum cost for a given number of stops.
    # A distances array `min_cost_with_stops[stops_taken][city]` is better for explicit Bellman-Ford
    # For BFS-based, it's a bit trickier, need to make sure we don't relax using
    # a path that's already worse given the stops.

    # Let's use a standard BFS but with a visited state that includes stops
    # min_cost_reached[city][stops_taken] stores min cost to reach city with exact stops_taken
    min_cost_reached = [
        [float("inf")] * (k + 2) for _ in range(n)
    ]  # k+1 for max stops, +1 for 0-stops
    min_cost_reached[src][0] = 0

    while queue:
        current_price, u, stops_taken = queue.popleft()

        # If we exceeded max stops, or this path is already worse than known, skip
        if stops_taken > k + 1:  # We allow K stops, meaning K+1 edges
            continue

        # If we are already at a cheaper price for 'u' with 'stops_taken' stops, continue
        # This check is crucial for correctness with costs.
        if current_price > min_cost_reached[u][stops_taken]:
            continue

        if u == dst:
            min_price = min(min_price, current_price)
            continue  # Found a path, continue searching for potentially cheaper paths

        # Explore neighbors
        for v, price_to_v in adj_list.get(u, []):
            new_price = current_price + price_to_v
            new_stops_taken = stops_taken + 1

            # Only consider if within K stops and if this path offers a cheaper cost
            if (
                new_stops_taken <= k + 1
                and new_price < min_cost_reached[v][new_stops_taken]
            ):
                min_cost_reached[v][new_stops_taken] = new_price
                queue.append((new_price, v, new_stops_taken))

    return min_price if min_price != float("inf") else -1


def find_smallest_set_of_vertices(n, edges):
    """
    Finds the smallest set of vertices from which all other nodes in a DAG are reachable.

    Args:
        n (int): The number of vertices.
        edges (list of lists): List of [u, v] directed edges.

    Returns:
        list: The smallest set of starting vertices.
    """
    # In a DAG, a node needs to be in the starting set only if no other node can reach it.
    # This means, nodes with an in-degree of 0 are the required starting points.

    in_degree = [0] * n
    for u, v in edges:
        in_degree[v] += 1

    result = []
    for i in range(n):
        if in_degree[i] == 0:
            result.append(i)

    return result


def min_reorder(n, connections):
    """
    Calculates the minimum number of road reorientations needed for all paths to lead to city 0.

    Args:
        n (int): The number of cities (nodes).
        connections (list of lists): List of [a, b] directed roads from a to b.

    Returns:
        int: The minimum number of reorientations.
    """
    # Create two adjacency lists: one for original directions, one for reversed (for traversal from 0)
    # adj[u] will store (v, is_original_direction)
    adj = collections.defaultdict(list)
    for u, v in connections:
        adj[u].append((v, 1))  # 1 indicates original direction (u -> v)
        adj[v].append((u, 0))  # 0 indicates reversed direction (v -> u)

    reorientations = 0
    visited = [False] * n

    # BFS to traverse from city 0
    queue = collections.deque([0])
    visited[0] = True

    while queue:
        u = queue.popleft()
        for v, is_original in adj[u]:
            if not visited[v]:
                visited[v] = True
                if (
                    is_original == 1
                ):  # If the original edge was u -> v, it points away from 0
                    reorientations += 1  # So, we need to reorient it
                queue.append(v)

    return reorientations
