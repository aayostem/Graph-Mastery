from collections import defaultdict, deque
import heapq


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Helper class for graph node cloning problems
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True
        return False


# -----------------------------------------------------------------------------
# 48. Cheapest Flights with K Stops (Dijkstra's with state)
# A modified Dijkstra's to find the cheapest path with a limit on stops.


def findCheapestPriceDijkstra(
    n: int, flights: list[list[int]], src: int, dst: int, k: int
) -> int:
    """
    Finds the cheapest price from a source to a destination with at most k stops
    using a modified Dijkstra's algorithm. The state in the priority queue is
    (price, node, stops).

    Time Complexity: O(E log(N*K)), as the size of the priority queue can be up to N*K.
    Space Complexity: O(N*K) for the distances and the priority queue.

    Args:
        n: The number of cities.
        flights: A list of [from, to, price] tuples.
        src: The source city.
        dst: The destination city.
        k: The maximum number of stops.

    Returns:
        The cheapest price, or -1 if the destination is unreachable.
    """
    adj = defaultdict(list)
    for u, v, p in flights:
        adj[u].append((v, p))

    pq = [(0, src, 0)]  # (price, node, stops)
    distances = {}  # Store (node, stops) -> min_price
    distances[(src, 0)] = 0

    while pq:
        price, u, stops = heapq.heappop(pq)

        if u == dst:
            return price

        if stops > k:
            continue

        for v, edge_price in adj[u]:
            new_price = price + edge_price
            if (v, stops + 1) not in distances or new_price < distances[(v, stops + 1)]:
                distances[(v, stops + 1)] = new_price
                heapq.heappush(pq, (new_price, v, stops + 1))

    return -1


# --- Example Usage ---
# flights8 = [[0,1,100],[1,2,100],[0,2,500]]
# print(f"Cheapest price (Dijkstra): {findCheapestPriceDijkstra(3, flights8, 0, 2, 1)}") # Expected: 200


# -----------------------------------------------------------------------------
# 50. Min Cut (Simplified with Max Flow)
# Finds the minimum number of edges to cut to separate two specified nodes.
def minCut(capacity_matrix: list[list[int]], s: int, t: int) -> int:
    """
    Calculates the size of the minimum cut of a flow network from source 's'
    to sink 't'. This is achieved by using the max-flow min-cut theorem, which
    states that the value of the max flow is equal to the capacity of the min cut.
    This solution uses the Edmonds-Karp algorithm for max flow.

    Time Complexity: O(V * E^2).
    Space Complexity: O(V^2).

    Args:
        capacity_matrix: A matrix representing edge capacities.
        s: The source node.
        t: The sink node.

    Returns:
        The value of the minimum cut.
    """
    num_nodes = len(capacity_matrix)
    residual_graph = [row[:] for row in capacity_matrix]
    max_flow = 0

    def bfs_path(parent):
        visited = [False] * num_nodes
        queue = deque([s])
        visited[s] = True
        parent[s] = -1

        while queue:
            u = queue.popleft()
            for v in range(num_nodes):
                if not visited[v] and residual_graph[u][v] > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u
        return visited[t]

    while bfs_path(parent := [-1] * num_nodes):
        path_flow = float("inf")
        v = t
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, residual_graph[u][v])
            v = u

        max_flow += path_flow

        v = t
        while v != s:
            u = parent[v]
            residual_graph[u][v] -= path_flow
            residual_graph[v][u] += path_flow
            v = u

    return max_flow


# --- Example Usage ---
# # Example capacity matrix from a flow network
# capacity_matrix_ex = [[0, 16, 13, 0, 0, 0], [0, 0, 10, 12, 0, 0], [0, 4, 0, 0, 14, 0],
#                       [0, 0, 9, 0, 0, 20], [0, 0, 0, 7, 0, 4], [0, 0, 0, 0, 0, 0]]
# print(f"Min cut capacity: {minCut(capacity_matrix_ex, 0, 5)}") # Expected: 23

# ------------------------------------------------------------------------------------------------------


# 41. Clone Graph
# Creates a deep copy of a graph.
def cloneGraph(node: "Node") -> "Node":
    """
    Creates a deep copy of an undirected graph. This is a graph traversal
    problem that can be solved with BFS or DFS, using a hash map to keep
    track of visited nodes and their clones to avoid cycles and redundant work.

    Time Complexity: O(V + E), where V is the number of vertices and E is the number of edges.
    Space Complexity: O(V) for the map and queue/stack.

    Args:
        node: The starting node of the graph.

    Returns:
        The cloned starting node.
    """
    if not node:
        return None

    cloned_map = {node: Node(node.val)}
    queue = deque([node])

    while queue:
        original_node = queue.popleft()
        clone_node = cloned_map[original_node]

        for neighbor in original_node.neighbors:
            if neighbor not in cloned_map:
                cloned_map[neighbor] = Node(neighbor.val)
                queue.append(neighbor)
            clone_node.neighbors.append(cloned_map[neighbor])

    return cloned_map[node]


# --- Example Usage ---
# node_41_1 = Node(1)
# node_41_2 = Node(2)
# node_41_3 = Node(3)
# node_41_4 = Node(4)
# node_41_1.neighbors.extend([node_41_2, node_41_4])
# node_41_2.neighbors.extend([node_41_1, node_41_3])
# node_41_3.neighbors.extend([node_41_2, node_41_4])
# node_41_4.neighbors.extend([node_41_1, node_41_3])
# cloned_node = cloneGraph(node_41_1)
# # print(f"Cloned node value: {cloned_node.val}") # Expected: 1


# -----------------------------------------------------------------------------
# 43. Course Schedule
# Determines if it's possible to finish all courses given prerequisites.
def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    """
    Given a list of courses and their prerequisites, determines if it is
    possible to finish all courses. This is a cycle detection problem in a
    directed graph. A topological sort will fail if a cycle exists.

    Time Complexity: O(V + E), where V is the number of courses and E is the number of prerequisites.
    Space Complexity: O(V + E) for the adjacency list and in-degree array.

    Args:
        numCourses: The total number of courses.
        prerequisites: A list of [course, prerequisite] pairs.

    Returns:
        True if all courses can be finished, False otherwise.
    """
    adj = [[] for _ in range(numCourses)]
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        adj[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    count = 0

    while queue:
        node = queue.popleft()
        count += 1

        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return count == numCourses


# --- Example Usage ---
# prerequisites_43 = [[1,0], [0,1]]
# print(f"Can finish courses: {canFinish(2, prerequisites_43)}") # Expected: False


# -----------------------------------------------------------------------------
# 44. Course Schedule II
# Returns one valid course order to finish all courses.
def findOrder(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    """
    Given a list of courses and their prerequisites, returns one valid order
    to take all courses. This is a topological sorting problem. If a valid
    order doesn't exist (due to a cycle), an empty list is returned.

    Time Complexity: O(V + E).
    Space Complexity: O(V + E).

    Args:
        numCourses: The number of courses.
        prerequisites: The list of prerequisites.

    Returns:
        A list of course numbers in a valid order, or an empty list.
    """
    adj = [[] for _ in range(numCourses)]
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        adj[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == numCourses else []


# --- Example Usage ---
# prerequisites_44 = [[1,0]]
# print(f"Course order: {findOrder(2, prerequisites_44)}") # Expected: [0, 1]


# -----------------------------------------------------------------------------
# 46. Walls and Gates
# Finds the shortest distance from each empty room to the nearest gate.
def wallsAndGates(rooms: list[list[int]]):
    """
    Given a 2D grid, fill each empty room with the shortest distance to a gate.
    This is a classic multi-source Breadth-First Search (BFS) problem. We start
    a BFS from all gates simultaneously and expand outwards.

    Time Complexity: O(R * C).
    Space Complexity: O(R * C) for the queue.

    Args:
        rooms: The 2D grid with gates (0), walls (-1), and empty rooms (inf).
    """
    if not rooms or not rooms[0]:
        return

    rows, cols = len(rooms), len(rooms[0])
    queue = deque()

    for r in range(rows):
        for c in range(cols):
            if rooms[r][c] == 0:
                queue.append((r, c, 0))  # (row, col, distance)

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        r, c, dist = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and rooms[nr][nc] == float("inf"):
                rooms[nr][nc] = dist + 1
                queue.append((nr, nc, dist + 1))


# --- Example Usage ---
# rooms_46 = [[float('inf'),-1,0,float('inf')],[float('inf'),float('inf'),float('inf'),-1],[float('inf'),-1,float('inf'),-1],[0,-1,float('inf'),float('inf')]]
# wallsAndGates(rooms_46)
# # Expected rooms: [[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]] (print to verify)
# # print(f"Rooms after processing: {rooms_46}")


# -----------------------------------------------------------------------------
# 47. Minimum Genetic Mutation
# Finds the shortest mutation path from a start gene to an end gene.
def minMutation(start: str, end: str, bank: list[str]) -> int:
    """
    Finds the minimum number of mutations to get from a start gene to an end
    gene, where each mutation must be a valid gene in a given bank. This is a
    shortest path problem on a graph where genes are nodes and edges exist
    between genes that differ by exactly one character.

    Time Complexity: O(L * B^2), where L is the length of the gene string and
                     B is the size of the bank, due to graph construction.
    Space Complexity: O(B) for the graph and queue.

    Args:
        start: The starting gene string.
        end: The target gene string.
        bank: A list of valid gene strings.

    Returns:
        The minimum number of mutations, or -1 if no such path exists.
    """
    bank_set = set(bank)
    if end not in bank_set:
        return -1

    queue = deque([(start, 0)])  # (gene, distance)
    visited = {start}

    while queue:
        current_gene, dist = queue.popleft()

        if current_gene == end:
            return dist

        for i in range(len(current_gene)):
            original_char = current_gene[i]
            for char in "ACGT":
                if char == original_char:
                    continue

                new_gene = current_gene[:i] + char + current_gene[i + 1 :]

                if new_gene in bank_set and new_gene not in visited:
                    visited.add(new_gene)
                    queue.append((new_gene, dist + 1))

    return -1


# --- Example Usage ---
# bank_47 = ["AACCGGTA", "AACCGCTA", "AAACGGTA"]
# print(f"Min mutations: {minMutation('AACCGGTT', 'AACCGGTA', bank_47)}") # Expected: 1


# -----------------------------------------------------------------------------
# 48. Rotting Oranges
# Finds the minimum time for all fresh oranges to rot.
def orangesRotting(grid: list[list[int]]) -> int:
    """
    Calculates the minimum time required for all fresh oranges to become rotten.
    This is a multi-source BFS problem where we start the traversal from all
    rotten oranges simultaneously.

    Time Complexity: O(R * C), where R and C are grid dimensions.
    Space Complexity: O(R * C) for the queue.

    Args:
        grid: A grid of 0 (empty), 1 (fresh), or 2 (rotten) oranges.

    Returns:
        The minimum time in minutes, or -1 if some oranges can never rot.
    """
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))  # (row, col, time)
            elif grid[r][c] == 1:
                fresh_count += 1

    max_time = 0
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        r, c, time = queue.popleft()
        max_time = max(max_time, time)

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2  # Rot the fresh orange
                fresh_count -= 1
                queue.append((nr, nc, time + 1))

    return max_time if fresh_count == 0 else -1


# --- Example Usage ---
# grid_48 = [[2,1,1],[1,1,0],[0,1,1]]
# print(f"Time for all oranges to rot: {orangesRotting(grid_48)}") # Expected: 4


# -----------------------------------------------------------------------------
# 50. Path with Maximum Probability
# Finds the path with the maximum product of probabilities from start to end.
def maxProbability(
    n: int, edges: list[list[int]], succProb: list[float], start: int, end: int
) -> float:
    """
    Finds the path with the maximum product of probabilities from a start node
    to an end node. This is a modified Dijkstra's algorithm using a max-heap.

    Time Complexity: O(E log V), where V is the number of nodes and E is the number of edges.
    Space Complexity: O(V + E) for the graph and the priority queue.

    Args:
        n: The number of nodes.
        edges: A list of [u, v] pairs.
        succProb: A list of success probabilities for each edge.
        start: The starting node.
        end: The target node.

    Returns:
        The maximum probability.
    """
    graph = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        graph[u].append((v, succProb[i]))
        graph[v].append((u, succProb[i]))

    max_probs = [0.0] * n
    max_probs[start] = 1.0

    pq = [(-1.0, start)]  # Max-heap stores (-probability, node)

    while pq:
        prob, u = heapq.heappop(pq)
        prob = -prob

        if prob < max_probs[u]:
            continue

        for v, p_edge in graph[u]:
            new_prob = prob * p_edge
            if new_prob > max_probs[v]:
                max_probs[v] = new_prob
                heapq.heappush(pq, (-new_prob, v))

    return max_probs[end]


# --- Example Usage ---
# n_50 = 3
# edges_50 = [[0,1],[1,2],[0,2]]
# succProb_50 = [0.5,0.5,0.2]
# print(f"Max probability: {maxProbability(n_50, edges_50, succProb_50, 0, 2)}") # Expected: 0.25
