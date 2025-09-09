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
