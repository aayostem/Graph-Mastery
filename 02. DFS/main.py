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
