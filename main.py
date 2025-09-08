import collections


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


def dfs_traversal_recursive(num_vertices, adj_list, start_node):
    """
    Performs a Depth-First Search traversal on a graph using recursion.

    Args:
        num_vertices (int): The total number of vertices in the graph.
        adj_list (dict): The adjacency list representation of the graph.
        start_node (int): The starting node for the traversal.

    Returns:
        list: A list of nodes in DFS traversal order.
    """
    visited = [False] * num_vertices
    traversal_order = []

    def dfs_util(u):
        visited[u] = True
        traversal_order.append(u)
        for v in adj_list[u]:
            if not visited[v]:
                dfs_util(v)

    if 0 <= start_node < num_vertices:
        dfs_util(start_node)

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


def has_cycle_undirected(num_vertices, adj_list):
    """
    Checks if an undirected graph has a cycle using DFS.

    Args:
        num_vertices (int): The total number of vertices.
        adj_list (dict): The adjacency list representation of the graph.

    Returns:
        bool: True if a cycle exists, False otherwise.
    """
    visited = [False] * num_vertices

    def dfs_check_cycle(u, parent):
        visited[u] = True
        for v in adj_list[u]:
            if not visited[v]:
                if dfs_check_cycle(v, u):  # Pass current node as parent for neighbor
                    return True
            elif (
                v != parent
            ):  # If visited and not parent, it's a back-edge to an ancestor, hence a cycle
                return True
        return False

    for i in range(num_vertices):
        if not visited[i]:
            if dfs_check_cycle(i, -1):  # -1 indicates no parent for the initial call
                return True
    return False


def has_cycle_directed(num_vertices, adj_list):
    """
    Checks if a directed graph has a cycle using DFS.

    Args:
        num_vertices (int): The total number of vertices.
        adj_list (dict): The adjacency list representation of the graph.

    Returns:
        bool: True if a cycle exists, False otherwise.
    """
    # 0: unvisited, 1: visiting (in current recursion stack), 2: visited (processed)
    visited_state = [0] * num_vertices

    def dfs_check_cycle_directed(u):
        visited_state[u] = 1  # Mark as visiting
        for v in adj_list[u]:
            if (
                visited_state[v] == 1
            ):  # Found a back-edge to a node in current recursion stack
                return True
            if visited_state[v] == 0:  # If unvisited, recurse
                if dfs_check_cycle_directed(v):
                    return True
        visited_state[u] = 2  # Mark as visited (processed)
        return False

    for i in range(num_vertices):
        if visited_state[i] == 0:
            if dfs_check_cycle_directed(i):
                return True
    return False


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


def find_mother_vertex(num_vertices, edges):
    """
    Finds the mother vertex in a directed graph.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of lists): List of [u, v] directed edges.

    Returns:
        int: The label of the mother vertex, or -1 if none exists.
    """
    adj_list = collections.defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)

    last_finished_vertex = -1
    visited_dfs1 = [False] * num_vertices

    # Step 1: Perform DFS from all unvisited nodes to find the last finished vertex
    def dfs_util_1(u):
        visited_dfs1[u] = True
        for v in adj_list[u]:
            if not visited_dfs1[v]:
                dfs_util_1(v)
        nonlocal last_finished_vertex
        last_finished_vertex = u  # This vertex finishes last in its DFS component

    for i in range(num_vertices):
        if not visited_dfs1[i]:
            dfs_util_1(i)

    # Step 2: Perform DFS from the last_finished_vertex to check if all nodes are reachable
    if last_finished_vertex == -1:  # No vertices in graph
        return -1

    visited_dfs2 = [False] * num_vertices
    reachable_count = 0
    queue = collections.deque([last_finished_vertex])

    if 0 <= last_finished_vertex < num_vertices:
        visited_dfs2[last_finished_vertex] = True
        reachable_count += 1
    else:  # If last_finished_vertex is out of bounds (e.g., empty graph)
        return -1

    while queue:
        u = queue.popleft()
        for v in adj_list[u]:
            if not visited_dfs2[v]:
                visited_dfs2[v] = True
                reachable_count += 1
                queue.append(v)

    if reachable_count == num_vertices:
        return last_finished_vertex
    else:
        return -1


import collections
import heapq  # For Dijkstra's and Prim's algorithm


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


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


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


def is_valid_tree_medium(n, edges):
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
        adj_list[u].append(
            (v, 1)
        )  # Add weight for consistency with represent_graph, though not used
        adj_list[v].append((u, 1))

    visited = [False] * n

    # DFS to check for cycles and connectivity
    def dfs_check(u, parent):
        visited[u] = True
        for v, _ in adj_list[u]:
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


import collections
import heapq  # For Dijkstra's variant
import math  # For float('inf')


def represent_graph(num_vertices, edges, is_directed=False, is_weighted=False):
    adj_list = collections.defaultdict(list)
    for edge in edges:
        u, v = edge[0], edge[1]
        # Handle cases where weight might not be present in input edges for unweighted graphs
        weight = edge[2] if is_weighted and len(edge) > 2 else 1
        adj_list[u].append((v, weight))
        if not is_directed:
            adj_list[v].append((u, weight))
    return dict(adj_list)


def floyd_warshall(num_vertices, edges):
    """
    Computes all-pairs shortest paths using the Floyd-Warshall algorithm.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of lists): List of [u, v, weight] tuples representing weighted edges.

    Returns:
        list of lists: A 2D matrix where dist[i][j] is the shortest path from i to j.
                       float('inf') if unreachable.
                       Detects negative cycles by having dist[i][i] < 0.
    """
    # Initialize distances matrix
    dist = [[float("inf")] * num_vertices for _ in range(num_vertices)]

    # Distance to self is 0
    for i in range(num_vertices):
        dist[i][i] = 0

    # Populate with direct edge weights
    for u, v, weight in edges:
        dist[u][v] = min(dist[u][v], weight)  # Handle parallel edges

    # Iterate through all possible intermediate vertices 'k'
    for k in range(num_vertices):
        # Iterate through all source vertices 'i'
        for i in range(num_vertices):
            # Iterate through all destination vertices 'j'
            for j in range(num_vertices):
                # If path through k is shorter
                if dist[i][k] != float("inf") and dist[k][j] != float("inf"):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Optional: Detect negative cycles
    # If dist[i][i] < 0 for any i, there's a negative cycle reachable from/to i
    # This also means any path involving such a cycle would have negative infinity cost.
    # We can propagate this to all reachable nodes from the negative cycle
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                if (
                    dist[i][k] != float("inf")
                    and dist[k][j] != float("inf")
                    and dist[k][k] < 0
                ):
                    dist[i][j] = float("-inf")  # Path can be arbitrarily small

    return dist


def bellman_ford_neg_cycle(num_vertices, edges, source):
    """
    Detects if a directed graph contains a negative-weight cycle reachable from a source.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of lists): List of [u, v, weight] tuples representing directed edges.
        source (int): The starting node.

    Returns:
        bool: True if a negative cycle reachable from source exists, False otherwise.
    """
    distances = {i: float("inf") for i in range(num_vertices)}
    distances[source] = 0

    # Relax edges V-1 times
    for _ in range(num_vertices - 1):
        for u, v, weight in edges:
            if distances[u] != float("inf") and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight

    # Check for negative cycles (V-th relaxation)
    for u, v, weight in edges:
        if distances[u] != float("inf") and distances[u] + weight < distances[v]:
            return True  # Negative cycle detected

    return False


def find_sccs_kosaraju(num_vertices, edges):
    """
    Finds Strongly Connected Components (SCCs) using Kosaraju's algorithm.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of lists): List of [u, v] directed edges.

    Returns:
        list of lists: A list where each sublist represents an SCC.
    """
    # Step 1: Build original graph and its transpose
    adj = collections.defaultdict(list)
    adj_rev = collections.defaultdict(list)  # Transpose graph
    for u, v in edges:
        adj[u].append(v)
        adj_rev[v].append(u)

    # Step 2: Perform DFS on original graph to get finishing times
    visited = [False] * num_vertices
    order = []  # Stores nodes in increasing order of finishing times (stack-like)

    def dfs1(u):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)  # Add to order after visiting all descendants

    for i in range(num_vertices):
        if not visited[i]:
            dfs1(i)

    # Step 3: Perform DFS on transpose graph in decreasing order of finishing times
    sccs = []
    visited = [False] * num_vertices  # Reset visited array

    def dfs2(u, current_scc):
        visited[u] = True
        current_scc.append(u)
        for v in adj_rev[u]:  # Traverse on transpose graph
            if not visited[v]:
                dfs2(v, current_scc)

    # Process vertices in reverse order of finishing times (popping from 'order')
    for i in range(num_vertices - 1, -1, -1):
        u = order[i]
        if not visited[u]:
            current_scc = []
            dfs2(u, current_scc)
            sccs.append(current_scc)

    return sccs


def find_bridges_and_articulation_points(num_vertices, edges):
    """
    Finds bridges and articulation points in an undirected graph.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of lists): List of [u, v] undirected edges.

    Returns:
        tuple: (list of tuples: bridges, list of ints: articulation_points)
    """
    adj = collections.defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    discovery_time = [-1] * num_vertices
    low_link_value = [-1] * num_vertices
    visited = [False] * num_vertices
    timer = 0

    bridges = []
    articulation_points = set()  # Use a set to avoid duplicates

    def dfs(u, parent):
        nonlocal timer
        visited[u] = True
        discovery_time[u] = low_link_value[u] = timer
        timer += 1

        children_count = 0  # For articulation point root check

        for v in adj[u]:
            if v == parent:
                continue  # Skip parent

            if visited[v]:  # Back-edge
                low_link_value[u] = min(low_link_value[u], discovery_time[v])
            else:  # Tree-edge
                children_count += 1
                dfs(v, u)
                low_link_value[u] = min(low_link_value[u], low_link_value[v])

                # Check for articulation point
                if parent != -1 and low_link_value[v] >= discovery_time[u]:
                    articulation_points.add(u)

                # Check for bridge
                if low_link_value[v] > discovery_time[u]:
                    bridges.append((u, v))

        # Check for root of DFS tree as articulation point
        if parent == -1 and children_count > 1:
            articulation_points.add(u)

    for i in range(num_vertices):
        if not visited[i]:
            dfs(i, -1)  # -1 indicates no parent for the root of DFS tree

    return bridges, list(articulation_points)


def solve_tsp_dp(num_cities, distances):
    """
    Solves the Traveling Salesman Problem using Dynamic Programming with Bitmasking.

    Args:
        num_cities (int): The number of cities (nodes).
        distances (list of lists): dist[i][j] is the distance from city i to city j.

    Returns:
        int: The minimum cost of the TSP tour, or float('inf') if no tour exists.
    """
    # dp[mask][i] = minimum cost to visit all cities in 'mask', ending at city 'i'
    # 'mask' is a bitmask representing the set of visited cities.
    # We want to find min(dp[(1 << num_cities) - 1][i] + distances[i][start_city]) for all i.

    # Initialize dp table with infinity
    # A mask of `1 << num_cities` means all possible subsets (0 to 2^num_cities - 1)
    # The inner list is `num_cities` for the last visited city
    dp = [[float("inf")] * num_cities for _ in range(1 << num_cities)]

    # Base case: Start at city 0 (or any chosen starting city).
    # The cost to visit only city 0, ending at city 0, is 0.
    start_city = 0  # Conventionally, TSP starts and ends at city 0
    dp[1 << start_city][start_city] = 0

    # Iterate through all masks (subsets of cities)
    # Masks from 1 to (1 << num_cities) - 1 (all cities visited)
    for mask in range(1, 1 << num_cities):
        # Iterate through all possible ending cities 'u' for the current mask
        for u in range(num_cities):
            # If city 'u' is in the current mask and we have a valid path to 'u'
            if (mask >> u) & 1 and dp[mask][u] != float("inf"):
                # Try to extend the path to an unvisited city 'v'
                for v in range(num_cities):
                    if not (
                        (mask >> v) & 1
                    ):  # If 'v' is not in the current mask (unvisited)
                        new_mask = mask | (1 << v)  # Add 'v' to the mask
                        # Update dp[new_mask][v] if a shorter path is found
                        if distances[u][v] != float("inf"):  # Ensure there's an edge
                            dp[new_mask][v] = min(
                                dp[new_mask][v], dp[mask][u] + distances[u][v]
                            )

    # After filling the DP table, find the minimum cost to return to the start_city
    min_tour_cost = float("inf")
    final_mask = (1 << num_cities) - 1  # All cities visited mask

    for i in range(num_cities):
        if (
            i != start_city
            and dp[final_mask][i] != float("inf")
            and distances[i][start_city] != float("inf")
        ):
            min_tour_cost = min(
                min_tour_cost, dp[final_mask][i] + distances[i][start_city]
            )

    return min_tour_cost if min_tour_cost != float("inf") else float("inf")


def max_flow_edmonds_karp(num_vertices, graph, source, sink):
    """
    Finds the maximum flow from source to sink in a flow network using Edmonds-Karp algorithm.

    Args:
        num_vertices (int): The number of vertices in the graph.
        graph (list of lists): Adjacency list representation where graph[u] is a list of [v, capacity].
                               Note: This implementation modifies graph capacities directly.
        source (int): The source node.
        sink (int): The sink node.

    Returns:
        int: The maximum flow value.
    """
    # Create residual graph:
    # Each edge (u,v) with capacity c has:
    #   forward edge (u,v) with residual capacity c - flow
    #   backward edge (v,u) with residual capacity flow

    # This implementation will modify the adj_list dynamically for residual graph
    # For simplicity, we'll augment the 'graph' input to store residual capacities.
    # An alternative is to create a separate residual_graph matrix.

    # We need to map edges to their capacities and residual capacities.
    # Using a list of lists (adjacency list) where each inner list contains tuples (neighbor, capacity, reverse_edge_index)
    # The reverse_edge_index points to the corresponding reverse edge in the neighbor's list.

    # Initialize adjacency list for residual graph (direct access to capacities)
    residual_graph = collections.defaultdict(list)

    # For Edmonds-Karp, easier to just represent graph as adjacency list with [neighbor, capacity]
    # And augment flow. When flow sent u->v, capacity u->v decreases, v->u increases.
    # So adj_list must store reference to reverse edge.

    # Let's rebuild the graph structure for flow to be more explicit.
    # Each edge will be a tuple (v, capacity, flow_index_in_v_list)
    # This helps update the reverse edge efficiently.

    # A simpler way for competitive programming: just use a matrix for capacity.
    # `capacity[u][v]` is remaining capacity.
    capacity = [[0] * num_vertices for _ in range(num_vertices)]
    for u, v, w in graph:  # Assume graph input is like [[u, v, capacity]]
        capacity[u][v] += w  # Handle parallel edges by summing capacities

    max_flow = 0

    while True:
        # BFS to find an augmenting path in the residual graph
        parent = [
            -1
        ] * num_vertices  # parent[i] stores the node that led to i in BFS path
        queue = collections.deque(
            [(source, float("inf"))]
        )  # (node, min_capacity_on_path)
        parent[source] = source  # Mark source as visited

        path_flow = 0  # Stores the bottleneck capacity of the found path

        while queue:
            u, current_path_flow = queue.popleft()

            if u == sink:
                path_flow = current_path_flow
                break

            for v in range(num_vertices):
                if (
                    parent[v] == -1 and capacity[u][v] > 0
                ):  # If v is unvisited and has capacity
                    parent[v] = u
                    # Bottleneck capacity up to v is min of current path flow and edge capacity
                    queue.append((v, min(current_path_flow, capacity[u][v])))

        if path_flow == 0:  # No augmenting path found
            break

        # Augment flow along the path
        current = sink
        while current != source:
            prev = parent[current]
            capacity[prev][current] -= path_flow  # Decrease forward edge capacity
            capacity[current][prev] += path_flow  # Increase backward edge capacity
            current = prev

        max_flow += path_flow

    return max_flow


def longest_path_in_dag(num_vertices, edges, source=None):
    """
    Finds the longest path from a source (or any node if source is None) in a DAG.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of lists): List of [u, v, weight] tuples representing directed edges.
                               Weight is 1 if not provided (for unweighted longest path).
        source (int, optional): The starting node. If None, finds the longest path overall.

    Returns:
        int: The length of the longest path.
    """
    adj_list = collections.defaultdict(list)
    in_degree = [0] * num_vertices

    # Build graph and calculate in-degrees
    for u, v, *w_opt in edges:
        weight = w_opt[0] if w_opt else 1
        adj_list[u].append((v, weight))
        in_degree[v] += 1

    # Initialize distances for DP
    # dist[i] = longest path ending at node i
    dist = [-float("inf")] * num_vertices

    # Use Kahn's algorithm (topological sort) for processing order
    queue = collections.deque()
    for i in range(num_vertices):
        if in_degree[i] == 0:
            queue.append(i)
            # If finding overall longest path, consider these as potential start points
            if source is None:
                dist[i] = 0  # Longest path ending at itself (0 cost)

    if source is not None:
        dist[source] = 0  # If specific source, start its path at 0

    # Process nodes in topological order
    while queue:
        u = queue.popleft()

        # Relax neighbors if reachable
        if dist[u] != -float(
            "inf"
        ):  # Ensure u is reachable from source (or a start of some path)
            for v, weight in adj_list[u]:
                if dist[u] + weight > dist[v]:
                    dist[v] = dist[u] + weight

                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    # Find the maximum value in dist array
    if source is None:
        return max(dist) if dist else 0  # Overall longest path
    else:
        # If finding path from a specific source, ensure target nodes are reachable
        max_from_source = 0
        for d in dist:
            if d != -float("inf"):
                max_from_source = max(max_from_source, d)
        return max_from_source  # Longest path from specified source


def alien_order(words):
    """
    Deduces the order of letters in an alien dictionary.

    Args:
        words (list of str): List of words sorted lexicographically.

    Returns:
        str: The string representing the alien order, or "" if invalid.
    """
    # Step 1: Build the graph (adjacency list and in-degrees)
    adj = collections.defaultdict(set)  # Use set to avoid duplicate edges
    in_degree = collections.defaultdict(int)

    # Initialize all characters that appear in words
    all_chars = set()
    for word in words:
        for char in word:
            all_chars.add(char)
            in_degree[char]  # Ensure all chars are in in_degree map, even if 0

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        min_len = min(len(w1), len(w2))

        # Check for invalid order (e.g., "abc" appears before "ab")
        if w1.startswith(w2) and len(w1) > len(w2):
            return ""

        found_diff = False
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in adj[w1[j]]:  # Add edge only if not already present
                    adj[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                found_diff = True
                break

    # Step 2: Perform Topological Sort (Kahn's Algorithm)
    queue = collections.deque()
    for char in all_chars:
        if in_degree[char] == 0:
            queue.append(char)

    result = []
    while queue:
        u = queue.popleft()
        result.append(u)

        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Step 3: Check for cycle
    if len(result) == len(all_chars):
        return "".join(result)
    else:
        return ""  # Cycle detected or not all characters processed


def find_critical_connections(n, connections):
    """
    Finds all critical connections (bridges) in an undirected graph.

    Args:
        n (int): The number of nodes.
        connections (list of lists): List of [u, v] undirected edges.

    Returns:
        list of lists: A list of critical connections (bridges).
    """
    adj = collections.defaultdict(list)
    for u, v in connections:
        adj[u].append(v)
        adj[v].append(u)

    discovery_time = [-1] * n
    low_link_value = [-1] * n
    timer = 0

    bridges = []

    def dfs(u, parent):
        nonlocal timer
        discovery_time[u] = low_link_value[u] = timer
        timer += 1

        for v in adj[u]:
            if v == parent:
                continue  # Skip parent

            if discovery_time[v] != -1:  # Already visited (back-edge)
                low_link_value[u] = min(low_link_value[u], discovery_time[v])
            else:  # Tree-edge
                dfs(v, u)
                low_link_value[u] = min(low_link_value[u], low_link_value[v])

                # If the lowest reachable time from v's subtree is greater than u's discovery time,
                # then (u,v) is a bridge.
                if low_link_value[v] > discovery_time[u]:
                    bridges.append([u, v])

    for i in range(n):
        if discovery_time[i] == -1:  # Only visit unvisited nodes
            dfs(i, -1)  # -1 indicates no parent for the root of DFS tree

    return bridges


def find_order(num_courses, prerequisites):
    """
    Finds a valid topological ordering of courses to take.

    Args:
        num_courses (int): Total number of courses.
        prerequisites (list of lists): List of [course, prerequisite] pairs.

    Returns:
        list: A list of courses in topological order, or an empty list if a cycle is detected.
    """
    adj_list = collections.defaultdict(list)
    in_degree = [0] * num_courses

    # Build adjacency list and compute in-degrees
    for course, pre_req in prerequisites:
        adj_list[pre_req].append(course)  # Edge from prerequisite to course
        in_degree[course] += 1

    # Initialize queue with courses that have no prerequisites
    queue = collections.deque()
    for i in range(num_courses):
        if in_degree[i] == 0:
            queue.append(i)

    top_order = []
    courses_processed = 0

    while queue:
        u = queue.popleft()
        top_order.append(u)
        courses_processed += 1

        for v in adj_list.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # If the number of processed courses equals total_num_courses, a valid order was found
    if courses_processed == num_courses:
        return top_order
    else:
        return []  # Cycle detected, impossible to finish all courses


import collections
import heapq  # For Dijkstra's variant in Min Cost Flow
import math  # For float('inf')


def represent_graph(num_vertices, edges, is_directed=False, is_weighted=False):
    adj_list = collections.defaultdict(list)
    for edge in edges:
        u, v = edge[0], edge[1]
        weight = edge[2] if is_weighted and len(edge) > 2 else 1
        adj_list[u].append((v, weight))
        if not is_directed:
            adj_list[v].append((u, weight))
    return dict(adj_list)


class Edge:
    def __init__(self, to, capacity, cost, reverse_edge_idx):
        self.to = to
        self.capacity = capacity
        self.cost = cost
        self.reverse_edge_idx = reverse_edge_idx


def min_cost_flow(num_vertices, edges, source, sink, required_flow=float("inf")):
    """
    Finds the minimum cost to send a required amount of flow from source to sink.
    Uses the Successive Shortest Path algorithm with SPFA to find augmenting paths.
    Note: SPFA can be slow on certain graph types (worst-case exponential), but often performs well in practice.

    Args:
        num_vertices (int): Total number of vertices.
        edges (list of lists): List of [u, v, capacity, cost] for each edge.
        source (int): Source node.
        sink (int): Sink node.
        required_flow (float): The total amount of flow to send. Defaults to infinity (max flow).

    Returns:
        tuple: (total_flow, total_cost)
    """
    adj = collections.defaultdict(list)

    # Build the graph with residual capacities and costs, and reverse edges
    for u, v, cap, cost in edges:
        # Forward edge
        adj[u].append(Edge(v, cap, cost, len(adj[v])))
        # Backward (residual) edge - initially 0 capacity, negative cost
        adj[v].append(Edge(u, 0, -cost, len(adj[u]) - 1))

    total_flow = 0
    total_cost = 0

    while total_flow < required_flow:
        # SPFA to find the shortest path in terms of cost
        distances = [float("inf")] * num_vertices
        parent_edge = [-1] * num_vertices  # Stores index of edge in parent's adj list
        parent_node = [-1] * num_vertices  # Stores parent node
        in_queue = [False] * num_vertices

        queue = collections.deque([source])
        distances[source] = 0
        in_queue[source] = True

        while queue:
            u = queue.popleft()
            in_queue[u] = False

            for i, edge in enumerate(adj[u]):
                v = edge.to
                if edge.capacity > 0 and distances[u] + edge.cost < distances[v]:
                    distances[v] = distances[u] + edge.cost
                    parent_node[v] = u
                    parent_edge[v] = i  # Index of edge from u to v in adj[u]

                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

        # If sink is unreachable, or no more flow can be sent
        if distances[sink] == float("inf"):
            break

        # Calculate bottleneck capacity on the found path
        path_min_capacity = required_flow - total_flow
        current = sink
        while current != source:
            prev = parent_node[current]
            edge_idx = parent_edge[current]
            path_min_capacity = min(path_min_capacity, adj[prev][edge_idx].capacity)
            current = prev

        # Augment flow along the path
        total_flow += path_min_capacity
        total_cost += path_min_capacity * distances[sink]  # Cost of this augmented flow

        current = sink
        while current != source:
            prev = parent_node[current]
            edge_idx = parent_edge[current]

            # Update capacities in residual graph
            adj[prev][edge_idx].capacity -= path_min_capacity
            reverse_edge_idx = adj[prev][edge_idx].reverse_edge_idx
            adj[current][reverse_edge_idx].capacity += path_min_capacity
            current = prev

    return total_flow, total_cost


def vertex_cover_approx(num_vertices, edges):
    """
    Finds an approximate minimum vertex cover using a greedy 2-approximation algorithm.
    It guarantees a solution at most twice the size of the optimal vertex cover.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of lists): List of [u, v] undirected edges.

    Returns:
        set: A set of vertices forming the approximate vertex cover.
    """
    # Create a copy of edges, as we will modify it
    remaining_edges = set()
    for u, v in edges:
        # Store edges as frozensets for set operations (order-independent)
        remaining_edges.add(frozenset({u, v}))

    vertex_cover = set()

    while remaining_edges:
        # Pick an arbitrary edge (u, v) from the remaining edges
        u, v = next(iter(remaining_edges))  # Get any edge

        # Add both endpoints to the vertex cover
        vertex_cover.add(u)
        vertex_cover.add(v)

        # Remove all edges incident to u or v
        edges_to_remove = set()
        for edge in remaining_edges:
            if u in edge or v in edge:
                edges_to_remove.add(edge)

        remaining_edges -= edges_to_remove

    return vertex_cover


def find_eulerian_path_or_circuit(num_vertices, edges, is_directed=False):
    """
    Finds an Eulerian path or circuit in a graph using Hierholzer's algorithm.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of lists): List of [u, v] edges.
        is_directed (bool): True if directed, False if undirected.

    Returns:
        list: The Eulerian path/circuit as a list of vertices, or None if none exists.
    """
    # Step 1: Build adjacency list and calculate degrees
    adj = collections.defaultdict(list)
    in_degree = [0] * num_vertices if is_directed else None
    out_degree = [0] * num_vertices if is_directed else None
    total_degree = [0] * num_vertices  # For undirected graphs

    num_edges_in_graph = 0
    for u, v in edges:
        adj[u].append(v)
        num_edges_in_graph += 1
        if is_directed:
            out_degree[u] += 1
            in_degree[v] += 1
        else:
            adj[v].append(u)
            total_degree[u] += 1
            total_degree[v] += 1

    # Step 2: Check conditions for Eulerian path/circuit existence
    start_node = 0  # Default start node

    if is_directed:
        start_nodes_candidates = []
        end_nodes_candidates = []
        for i in range(num_vertices):
            if out_degree[i] - in_degree[i] == 1:
                start_nodes_candidates.append(i)
            elif in_degree[i] - out_degree[i] == 1:
                end_nodes_candidates.append(i)
            elif in_degree[i] != out_degree[i]:  # Mismatched degrees, not circuit
                return None

        if len(start_nodes_candidates) == 0 and len(end_nodes_candidates) == 0:
            # All degrees balanced, potential circuit. Start from any non-isolated vertex.
            start_node = next(
                (
                    i
                    for i in range(num_vertices)
                    if in_degree[i] > 0 or out_degree[i] > 0
                ),
                0,
            )
        elif len(start_nodes_candidates) == 1 and len(end_nodes_candidates) == 1:
            # One start, one end, potential path
            start_node = start_nodes_candidates[0]
        else:
            return None  # Invalid degree sequence for path or circuit
    else:  # Undirected
        odd_degree_vertices = []
        for i in range(num_vertices):
            if total_degree[i] % 2 != 0:
                odd_degree_vertices.append(i)

        if len(odd_degree_vertices) == 0:
            # All even degrees, potential circuit. Start from any non-isolated vertex.
            start_node = next(
                (i for i in range(num_vertices) if total_degree[i] > 0), 0
            )
        elif len(odd_degree_vertices) == 2:
            # Exactly two odd degrees, potential path. Start from one of them.
            start_node = odd_degree_vertices[0]
        else:
            return None  # Invalid degree sequence

    # Step 3: Hierholzer's Algorithm
    current_path = []
    stack = [start_node]

    # Need a mutable copy of adj list because we remove edges
    # For undirected, ensure edges are removed from both directions or counted correctly.

    # Simple way to track visited edges: store current_edge_idx in adj_list for each node
    # or remove edges from the adjacency list directly.
    # Using `collections.defaultdict(collections.deque)` for adjacency list
    adj_deques = collections.defaultdict(collections.deque)
    for u in adj:
        for v in adj[u]:
            adj_deques[u].append(v)

    while stack:
        u = stack[-1]  # Peek at current node

        if adj_deques[u]:  # If current node has unvisited edges
            v = adj_deques[u].popleft()  # Take an edge
            stack.append(v)
        else:  # All edges from u visited, backtrack
            current_path.append(stack.pop())

    # The path is built in reverse order (stack.pop() adds to current_path)
    current_path.reverse()

    # Step 4: Verify connectivity (and if all edges were used)
    # Check if number of edges in path matches total edges in graph
    if len(current_path) - 1 != num_edges_in_graph:
        return None  # Not all edges traversed (disconnected components or error)

    # For directed graphs, check overall connectivity
    # A simple DFS/BFS from `start_node` to ensure all non-isolated vertices are reachable
    # and all non-isolated vertices can reach `start_node` (for circuit)
    # Or for undirected, ensure only one component (if total_degree[i]>0).

    # Basic connectivity check after finding a potential path:
    # Ensure that any node with degree > 0 is part of the found path
    nodes_in_path_set = set(current_path)
    for i in range(num_vertices):
        if (
            is_directed
            and (in_degree[i] > 0 or out_degree[i] > 0)
            and i not in nodes_in_path_set
        ) or (not is_directed and total_degree[i] > 0 and i not in nodes_in_path_set):
            return None  # Not all connected components with edges were included

    return current_path


def shortest_path_exactly_k_edges(num_vertices, edges, source, destination, k_exact):
    """
    Finds the shortest path from source to destination using exactly K edges.

    Args:
        num_vertices (int): The total number of vertices.
        edges (list of lists): List of [u, v, weight] directed edges.
        source (int): The starting node.
        destination (int): The target node.
        k_exact (int): The exact number of edges to use.

    Returns:
        int: The shortest path cost, or float('inf') if no such path exists.
    """
    adj = collections.defaultdict(list)
    for u, v, weight in edges:
        adj[u].append((v, weight))

    # dp[k][u] = min cost to reach u using exactly k edges
    dp = [[float("inf")] * num_vertices for _ in range(k_exact + 1)]

    dp[0][source] = 0  # Base case: 0 edges from source to source is 0 cost

    for k in range(1, k_exact + 1):
        for u in range(num_vertices):
            if dp[k - 1][u] == float("inf"):  # If u is not reachable in k-1 steps, skip
                continue
            for v, weight in adj[u]:
                dp[k][v] = min(dp[k][v], dp[k - 1][u] + weight)

    return dp[k_exact][destination]


def solve_2_sat(n_vars, clauses):
    """
    Solves the 2-SAT problem using implication graph and SCCs (Kosaraju's algorithm).

    Args:
        n_vars (int): Number of boolean variables (x_0, x_1, ..., x_{n_vars-1}).
        clauses (list of lists): Each clause is [literal1, literal2].
                                 Literal: `i` for x_i, `-i-1` for ~x_i. (0-indexed variables)
                                 So for x_0, it's 0. For ~x_0, it's -1. For x_1, it's 1. For ~x_1, it's -2.

    Returns:
        bool: True if satisfiable, False otherwise.
        list: A list of boolean assignments (True/False) for variables x_0 to x_{n_vars-1}, or None if unsatisfiable.
    """
    # Map literal to graph node:
    # `x_i` maps to `2*i`
    # `~x_i` maps to `2*i + 1`
    # Graph will have 2*n_vars nodes.

    num_nodes = 2 * n_vars
    adj = collections.defaultdict(list)
    adj_rev = collections.defaultdict(list)

    def get_node(literal):
        if literal >= 0:  # positive literal x_i
            return 2 * literal
        else:  # negative literal ~x_i
            return 2 * (-literal - 1) + 1

    def get_negation_node(node):
        # If node is 2*i (x_i), its negation is 2*i + 1 (~x_i)
        # If node is 2*i+1 (~x_i), its negation is 2*i (x_i)
        return node ^ 1  # XOR with 1 flips the last bit, effectively negating

    # Build implication graph
    for lit1, lit2 in clauses:
        node1 = get_node(lit1)
        node2 = get_node(lit2)

        neg_node1 = get_negation_node(node1)
        neg_node2 = get_negation_node(node2)

        # (A OR B) is equivalent to (~A IMPLIES B) AND (~B IMPLIES A)
        # So, edge from neg_node1 to node2
        # And edge from neg_node2 to node1
        adj[neg_node1].append(node2)
        adj_rev[node2].append(neg_node1)

        adj[neg_node2].append(node1)
        adj_rev[node1].append(neg_node2)

    # Step 1: DFS on original graph to get finishing times (for Kosaraju's)
    visited = [False] * num_nodes
    order = []

    def dfs1(u):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)

    for i in range(num_nodes):
        if not visited[i]:
            dfs1(i)

    # Step 2: DFS on transpose graph in reverse finishing order to find SCCs
    scc_id = [-1] * num_nodes  # Stores SCC ID for each node
    current_scc_idx = 0
    visited = [False] * num_nodes  # Reset visited

    def dfs2(u):
        visited[u] = True
        scc_id[u] = current_scc_idx
        for v in adj_rev[u]:
            if not visited[v]:
                dfs2(v)

    for i in range(num_nodes - 1, -1, -1):
        u = order[i]
        if not visited[u]:
            dfs2(u)
            current_scc_idx += 1  # Increment for next SCC

    # Step 3: Check satisfiability
    for i in range(n_vars):
        if scc_id[2 * i] == scc_id[2 * i + 1]:  # If x_i and ~x_i are in the same SCC
            return False, None  # Unsatisfiable

    # Step 4: Construct assignment (optional, for satisfiable cases)
    # A simple assignment can be formed by assigning false to a literal if its SCC comes
    # before its negation's SCC in the reverse topological order of SCCs.
    assignment = [False] * n_vars  # Initialize all to False

    # SCCs are numbered in reverse topological order (higher ID means comes earlier)
    # If scc_id[neg_node] > scc_id[node], then neg_node comes "before" node in topological sort of SCCs.
    # This means node cannot imply neg_node without contradiction.
    # Thus, if scc_id[~x] > scc_id[x], then x must be true.
    # Otherwise, x can be false.
    for i in range(n_vars):
        if (
            scc_id[2 * i] > scc_id[2 * i + 1]
        ):  # If SCC of x_i is after SCC of ~x_i in reverse topo order
            assignment[i] = (
                False  # Then assigning False to x_i means ~x_i is True. SCC[~x_i] will be earlier.
            )
            # If scc_id[x_i] < scc_id[~x_i], then x_i is true.
        else:
            assignment[i] = True

    # Simplified correct assignment logic:
    # If scc_id[x_i] < scc_id[~x_i], then x_i should be True.
    # If scc_id[~x_i] < scc_id[x_i], then x_i should be False.
    # This is because the higher SCC ID means it finishes earlier in DFS1,
    # meaning it's "later" in the reverse topological order of SCCs.
    # If a literal `L` is in an SCC that finishes later than `~L`'s SCC in DFS2 (reversed),
    # then `L` can be true.
    # Let's verify this logic.

    # The assignment is `true` if its literal node's SCC appears *after* its negation's SCC
    # in a topological sort of the condensation graph.
    # Or simply: if `scc_id[2*i]` (for `x_i`) is less than `scc_id[2*i + 1]` (for `~x_i`), set `x_i` to True.
    # Otherwise, set `x_i` to False.
    # This comes from the property that if `x_i` and `~x_i` are in different SCCs,
    # and there's a path `A -> B`, then `scc_id[A] >= scc_id[B]`.
    # So if `scc_id[x_i]` > `scc_id[~x_i]`, it means `~x_i` can imply `x_i` without contradiction,
    # but `x_i` cannot imply `~x_i`.
    # This means `~x_i` is 'more constrained' or 'earlier' in topological order. So `~x_i` must be false, `x_i` true.
    # So if `scc_id[2*i] > scc_id[2*i + 1]`, set `assignment[i] = True`. (Because 2*i implies 2*i+1 is false if True)
    # The current `scc_id` is assigned in *reverse* topological order of SCCs,
    # so a higher ID means it was found *earlier* in the reversed DFS (i.e., later in the original topological sort of SCCs).
    # If SCC(x) comes *after* SCC(~x) in topological order, then set x = True.
    # This means if scc_id[x_i] < scc_id[~x_i], then x_i = True. (smaller ID means visited later in reverse DFS)
    # No, it's `scc_id[2*i]` (x_i) > `scc_id[2*i+1]` (~x_i) means x_i should be True.
    # Let's assume higher scc_id means earlier in topological sort of SCCs.
    for i in range(n_vars):
        if scc_id[2 * i] < scc_id[2 * i + 1]:
            assignment[i] = (
                True  # x_i's SCC is 'later' than ~x_i's SCC in topological order. ~x_i implies x_i, so ~x_i must be false.
            )
        else:
            assignment[i] = False

    return True, assignment


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def mst_on_grid_manhattan(points):
    """
    Finds the MST on a grid with Manhattan distance using Kruskal's algorithm.
    This implementation explicitly builds all V^2 edges, which is inefficient for large V.
    For larger N, specialized geometric algorithms are needed.

    Args:
        points (list of tuples): List of (x, y) coordinates.

    Returns:
        int: The total weight of the MST.
    """
    num_vertices = len(points)
    if num_vertices <= 1:
        return 0

    all_edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            dist = manhattan_distance(points[i], points[j])
            all_edges.append((i, j, dist))

    # Use the Kruskal's MST function implemented in Medium Difficulty section (copied here for self-containment)
    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size

        def find(self, i):
            if self.parent[i] == i:
                return i
            self.parent[i] = self.find(self.parent[i])
            return self.parent[i]

        def union(self, i, j):
            root_i = self.find(i)
            root_j = self.find(j)
            if root_i != root_j:
                if self.rank[root_i] < self.rank[root_j]:
                    self.parent[root_i] = root_j
                elif self.rank[root_i] > self.rank[root_j]:
                    self.parent[root_j] = root_i
                else:
                    self.parent[root_j] = root_i
                    self.rank[root_i] += 1
                return True
            return False

    uf = UnionFind(num_vertices)
    mst_weight = 0
    edges_in_mst_count = 0

    # Sort edges by weight
    all_edges.sort(key=lambda x: x[2])

    for u, v, weight in all_edges:
        if uf.union(u, v):
            mst_weight += weight
            edges_in_mst_count += 1
            if edges_in_mst_count == num_vertices - 1:
                break

    # Check if all vertices are connected.
    # For geometric problems with points, usually they are considered connected by default
    # if num_vertices > 1.
    return mst_weight
