from collections import defaultdict, deque
import heapq


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


grid1 = [
    ["1", "1", "0", "0", "0"],
    ["1", "1", "0", "0", "0"],
    ["0", "0", "1", "0", "0"],
    ["0", "0", "0", "1", "1"],
]


def numIslands(grid: list[list[str]]) -> int:
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    islands_count = 0

    def dfs(r, c):

        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == "0":
            return

        grid[r][c] = "0"

        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):

            if grid[r][c] == "1":
                islands_count += 1

                dfs(r, c)

    return islands_count


def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    adj = [[] for _ in range(numCourses)]
    for course, prereq in prerequisites:
        adj[prereq].append(course)

    visited = [0] * numCourses

    def dfs(course):
        """DFS to detect a cycle."""
        if visited[course] == 1:
            return False  # Found a cycle
        if visited[course] == 2:
            return True  # Already visited and no cycle found

        visited[course] = 1  # Mark as visiting

        for next_course in adj[course]:
            if not dfs(next_course):
                return False

        visited[course] = 2  # Mark as fully visited
        return True

    for i in range(numCourses):
        if not dfs(i):
            return False

    return True


def solve(board: list[list[str]]) -> None:
    if not board or not board[0]:
        return

    rows, cols = len(board), len(board[0])

    def dfs(r, c):
        """Recursively marks all border-connected 'O's with a temporary marker 'E'."""
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != "O":
            return

        board[r][c] = "E"
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    # Phase 1: Run DFS from all 'O's on the border
    for r in range(rows):
        # Top and bottom borders
        if board[r][0] == "O":
            dfs(r, 0)
        if board[r][cols - 1] == "O":
            dfs(r, cols - 1)
    for c in range(cols):
        # Left and right borders
        if board[0][c] == "O":
            dfs(0, c)
        if board[rows - 1][c] == "O":
            dfs(rows - 1, c)

    # Phase 2: Flip the remaining 'O's and restore the marked 'O's
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == "O":
                board[r][c] = "X"
            elif board[r][c] == "E":
                board[r][c] = "O"


def pacificAtlantic(heights: list[list[int]]) -> list[list[int]]:
    if not heights or not heights[0]:
        return []

    rows, cols = len(heights), len(heights[0])
    pacific_reachable = [[False] * cols for _ in range(rows)]
    atlantic_reachable = [[False] * cols for _ in range(rows)]
    result = []

    def dfs(r, c, reachable):
        if reachable[r][c]:
            return
        reachable[r][c] = True

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and heights[nr][nc] >= heights[r][c]:
                dfs(nr, nc, reachable)

    for r in range(rows):
        dfs(r, 0, pacific_reachable)
    for c in range(cols):
        dfs(0, c, pacific_reachable)

    for r in range(rows):
        dfs(r, cols - 1, atlantic_reachable)
    for c in range(cols):
        dfs(rows - 1, c, atlantic_reachable)

    for r in range(rows):
        for c in range(cols):
            if pacific_reachable[r][c] and atlantic_reachable[r][c]:
                result.append([r, c])

    return result


def allPathsSourceTarget(graph: list[list[int]]) -> list[list[int]]:
    target = len(graph) - 1
    result = []

    def dfs(current_node, current_path):
        current_path.append(current_node)

        if current_node == target:
            result.append(list(current_path))

        for neighbor in graph[current_node]:
            dfs(neighbor, current_path)

        current_path.pop()

    dfs(0, [])
    return result
