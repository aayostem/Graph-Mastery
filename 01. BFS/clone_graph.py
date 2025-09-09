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
