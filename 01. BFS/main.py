from collections import defaultdict, deque
import heapq


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def cloneGraph(node: "Node") -> "Node":
    if not node:
        return None

    cloned = {node.val: Node(node.val)}
    queue = deque([node])

    while queue:
        original_node = queue.popleft()
        cloned_node = cloned[original_node.val]

        for neighbor in original_node.neighbors:

            if neighbor.val not in cloned:
                cloned[neighbor.val] = Node(neighbor.val)
                queue.append(neighbor)

            cloned_node.neighbors.append(cloned[neighbor.val])

    return cloned[node.val]


def findJudge(n: int, trust: list[list[int]]) -> int:

    in_degree = [0] * (n + 1)
    out_degree = [0] * (n + 1)

    for a, b in trust:
        out_degree[a] += 1
        in_degree[b] += 1

    for i in range(1, n + 1):

        if out_degree[i] == 0 and in_degree[i] == n - 1:
            return i

    return -1


def updateMatrix(mat: list[list[int]]) -> list[list[int]]:
    rows, cols = len(mat), len(mat[0])
    queue = deque()
    distances = [[-1] * cols for _ in range(rows)]

    # Initialize the queue with all '0' cells and set their distance to 0
    for r in range(rows):
        for c in range(cols):
            if mat[r][c] == 0:
                queue.append((r, c))
                distances[r][c] = 0

    # Run BFS
    while queue:
        r, c = queue.popleft()

        # Explore all four neighbors
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            # If the neighbor is within bounds and has not been visited yet
            if 0 <= nr < rows and 0 <= nc < cols and distances[nr][nc] == -1:
                distances[nr][nc] = distances[r][c] + 1
                queue.append((nr, nc))

    return distances


def wallsAndGates(rooms: list[list[int]]) -> None:
    if not rooms or not rooms[0]:
        return

    rows, cols = len(rooms), len(rooms[0])
    queue = deque()

    for r in range(rows):
        for c in range(cols):
            if rooms[r][c] == 0:
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and rooms[nr][nc] == 2147483647:
                rooms[nr][nc] = rooms[r][c] + 1
                queue.append((nr, nc))


def orangesRotting(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    fresh_count = 0
    queue = deque()

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                fresh_count += 1
            elif grid[r][c] == 2:
                queue.append((r, c, 0))

    if fresh_count == 0:
        return 0

    minutes = 0
    while queue:
        r, c, minutes = queue.popleft()

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh_count -= 1
                queue.append((nr, nc, minutes + 1))

    return minutes if fresh_count == 0 else -1


def dijkstra(graph, start_node):
    distances = {node: float("inf") for node in graph}
    distances[start_node] = 0

    priority_queue = [(0, start_node)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances


def bellmanFord(edges, num_nodes, start_node):
    distances = [float("inf")] * num_nodes
    distances[start_node] = 0

    for _ in range(num_nodes - 1):
        for u, v, weight in edges:
            if distances[u] != float("inf") and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight

    for u, v, weight in edges:
        if distances[u] != float("inf") and distances[u] + weight < distances[v]:
            return False

    return distances


def findCheapestPrice(
    n: int, flights: list[list[int]], src: int, dst: int, k: int
) -> int:
    adj = [[] for _ in range(n)]
    for u, v, w in flights:
        adj[u].append((v, w))

    distances = [float("inf")] * n
    distances[src] = 0

    queue = deque([(src, 0)])

    for i in range(k + 1):

        temp_distances = list(distances)

        for _ in range(len(queue)):
            u, stops = queue.popleft()

            for v, w in adj[u]:
                if distances[u] + w < temp_distances[v]:
                    temp_distances[v] = distances[u] + w
                    queue.append((v, stops + 1))

        distances = temp_distances

    return distances[dst] if distances[dst] != float("inf") else -1


def networkDelayTime(times: list[list[int]], n: int, k: int) -> int:
    graph = [[] for _ in range(n + 1)]
    for u, v, w in times:
        graph[u].append((v, w))

    distances = {i: float("inf") for i in range(1, n + 1)}
    distances[k] = 0

    pq = [(0, k)]

    while pq:
        curr_time, curr_node = heapq.heappop(pq)

        if curr_time > distances[curr_node]:
            continue

        for neighbor, travel_time in graph[curr_node]:
            if curr_time + travel_time < distances[neighbor]:
                distances[neighbor] = curr_time + travel_time
                heapq.heappush(pq, (distances[neighbor], neighbor))

    max_time = 0
    for time in distances.values():
        if time == float("inf"):
            return -1
        max_time = max(max_time, time)

    return max_time


def maxProbability(
    n: int, edges: list[list[int]], succProb: list[float], start: int, end: int
) -> float:
    graph = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        graph[u].append((v, succProb[i]))
        graph[v].append((u, succProb[i]))

    max_probs = [0.0] * n
    max_probs[start] = 1.0

    pq = [(-1.0, start)]

    while pq:
        prob, u = heapq.heappop(pq)
        prob = -prob

        if u == end:
            return prob

        if prob < max_probs[u]:
            continue

        for v, p_edge in graph[u]:
            new_prob = prob * p_edge
            if new_prob > max_probs[v]:
                max_probs[v] = new_prob
                heapq.heappush(pq, (-new_prob, v))

    return max_probs[end]


def findTheCity(n: int, edges: list[list[int]], distanceThreshold: int) -> int:
    distances = [[float("inf")] * n for _ in range(n)]
    for i in range(n):
        distances[i][i] = 0

    for u, v, w in edges:
        distances[u][v] = w
        distances[v][u] = w

    for k in range(n):
        for i in range(n):
            for j in range(n):
                distances[i][j] = min(
                    distances[i][j], distances[i][k] + distances[k][j]
                )

    min_neighbors = n
    result_city = -1

    for i in range(n):
        neighbors_count = 0
        for j in range(n):
            if i != j and distances[i][j] <= distanceThreshold:
                neighbors_count += 1

        if neighbors_count <= min_neighbors:
            min_neighbors = neighbors_count
            result_city = i

    return result_city


def floydWarshall(graph_matrix):
    n = len(graph_matrix)
    distances = [row[:] for row in graph_matrix]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                distances[i][j] = min(
                    distances[i][j], distances[i][k] + distances[k][j]
                )

    return distances


def countPaths(n: int, roads: list[list[int]]) -> int:
    MOD = 10**9 + 7
    graph = [[] for _ in range(n)]
    for u, v, t in roads:
        graph[u].append((v, t))
        graph[v].append((u, t))

    distances = [float("inf")] * n
    ways = [0] * n

    distances[0] = 0
    ways[0] = 1

    pq = [(0, 0)]

    while pq:
        dist, u = heapq.heappop(pq)

        if dist > distances[u]:
            continue

        for v, t in graph[u]:
            if dist + t < distances[v]:
                distances[v] = dist + t
                ways[v] = ways[u]
                heapq.heappush(pq, (distances[v], v))
            elif dist + t == distances[v]:
                ways[v] = (ways[v] + ways[u]) % MOD

    return ways[n - 1]


def shortestPathAllKeys(grid: list[str]) -> int:
    rows, cols = len(grid), len(grid[0])
    start_pos = None
    all_keys = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "@":
                start_pos = (r, c)
            elif "a" <= grid[r][c] <= "f":
                all_keys |= 1 << (ord(grid[r][c]) - ord("a"))

    queue = deque([(start_pos[0], start_pos[1], 0, 0)])
    visited = {(start_pos[0], start_pos[1], 0)}

    while queue:
        r, c, keys_mask, steps = queue.popleft()

        if keys_mask == all_keys:
            return steps

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "#"):
                continue

            cell = grid[nr][nc]
            new_keys_mask = keys_mask

            if "a" <= cell <= "f":
                new_keys_mask |= 1 << (ord(cell) - ord("a"))
            elif "A" <= cell <= "F":
                if not (keys_mask & (1 << (ord(cell) - ord("A")))):
                    continue

            if (nr, nc, new_keys_mask) not in visited:
                visited.add((nr, nc, new_keys_mask))
                queue.append((nr, nc, new_keys_mask, steps + 1))

    return -1


def swimInWater(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])

    pq = [(grid[0][0], 0, 0)]
    visited = set([(0, 0)])

    while pq:
        time, r, c = heapq.heappop(pq)

        if r == rows - 1 and c == cols - 1:
            return time

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                visited.add((nr, nc))
                new_time = max(time, grid[nr][nc])
                heapq.heappush(pq, (new_time, nr, nc))

    return -1
