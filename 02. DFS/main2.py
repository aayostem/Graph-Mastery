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


# 41. Word Search (Boggle)
# A classic backtracking problem on a grid.
def exist(board: list[list[str]], word: str) -> bool:
    """
    Checks if a word exists in a 2D grid of characters, where the word can be
    constructed from adjacent letters (horizontally or vertically). This is a
    backtracking problem solved with Depth-First Search (DFS).

    Time Complexity: O(R * C * 3^L), where R and C are grid dimensions and L is the word length.
                     The branching factor is at most 4, but we can't revisit the
                     same cell immediately, so it's closer to 3.
    Space Complexity: O(L) for the recursion stack.

    Args:
        board: The 2D grid of characters.
        word: The word to search for.

    Returns:
        True if the word exists, False otherwise.
    """
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, index):
        """Recursive backtracking function to find the word."""
        if index == len(word):
            return True

        if not (0 <= r < rows and 0 <= c < cols and board[r][c] == word[index]):
            return False

        # Mark the current cell as visited by changing its value
        temp = board[r][c]
        board[r][c] = "#"

        # Explore neighbors
        res = (
            backtrack(r + 1, c, index + 1)
            or backtrack(r - 1, c, index + 1)
            or backtrack(r, c + 1, index + 1)
            or backtrack(r, c - 1, index + 1)
        )

        # Restore the cell for other paths (backtracking)
        board[r][c] = temp
        return res

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == word[0] and backtrack(r, c, 0):
                return True

    return False


# --- Example Usage ---
# board1 = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# print(f"Word 'ABCCED' exists: {exist(board1, 'ABCCED')}") # Expected: True


# -----------------------------------------------------------------------------
# 42. Longest Increasing Path in a Matrix
# Finds the length of the longest increasing path in a matrix.
def longestIncreasingPath(matrix: list[list[int]]) -> int:
    """
    Finds the length of the longest increasing path in a matrix. This problem
    can be solved using memoization (dynamic programming) with DFS.

    Time Complexity: O(R * C), where R and C are the matrix dimensions, as each
                     cell is visited and computed only once.
    Space Complexity: O(R * C) for the memoization cache and recursion stack.

    Args:
        matrix: The input matrix of integers.

    Returns:
        The length of the longest increasing path.
    """
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    memo = {}

    def dfs(r, c):
        if (r, c) in memo:
            return memo[(r, c)]

        max_path = 1
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] > matrix[r][c]:
                max_path = max(max_path, 1 + dfs(nr, nc))

        memo[(r, c)] = max_path
        return max_path

    longest_path = 0
    for r in range(rows):
        for c in range(cols):
            longest_path = max(longest_path, dfs(r, c))

    return longest_path


# --- Example Usage ---
# matrix2 = [[9, 9, 4], [6, 6, 8], [2, 1, 1]]
# print(f"Longest increasing path length: {longestIncreasingPath(matrix2)}") # Expected: 4


# -----------------------------------------------------------------------------
# 43. Number of Connected Components in an Undirected Graph
# Counts the number of connected components in a graph.
def countComponents(n: int, edges: list[list[int]]) -> int:
    """
    Counts the number of connected components in an undirected graph. This can
    be solved by traversing the graph (with DFS or BFS) and counting how many
    times a new traversal needs to be started.

    Time Complexity: O(V + E), where V is the number of vertices and E is the number of edges.
    Space Complexity: O(V + E) for the adjacency list and visited set.

    Args:
        n: The number of nodes.
        edges: The list of edges.

    Returns:
        The number of connected components.
    """
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    count = 0

    def dfs(node):
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor)

    for i in range(n):
        if i not in visited:
            count += 1
            dfs(i)

    return count


# --- Example Usage ---
# edges3 = [[0, 1], [1, 2], [3, 4]]
# print(f"Number of connected components: {countComponents(5, edges3)}") # Expected: 2


# -----------------------------------------------------------------------------
# 44. Detect Cycle in a Directed Graph
# A DFS-based algorithm to detect a cycle in a directed graph.
def hasCycleDirected(num_nodes: int, edges: list[list[int]]) -> bool:
    """
    Detects if a directed graph contains a cycle. This is solved with a DFS-based
    approach using two visited arrays: one for the entire graph and one for the
    current recursion path.

    Time Complexity: O(V + E).
    Space Complexity: O(V + E) for the adjacency list and visited arrays.

    Args:
        num_nodes: The number of nodes.
        edges: The list of directed edges.

    Returns:
        True if a cycle exists, False otherwise.
    """
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    visited = [False] * num_nodes
    rec_stack = [False] * num_nodes

    def dfs(node):
        visited[node] = True
        rec_stack[node] = True

        for neighbor in adj[node]:
            if not visited[neighbor]:
                if dfs(neighbor):
                    return True
            elif rec_stack[neighbor]:
                return True  # Found a back edge to a node in the current path

        rec_stack[node] = False
        return False

    for i in range(num_nodes):
        if not visited[i]:
            if dfs(i):
                return True

    return False


# --- Example Usage ---
# edges4 = [[0, 1], [1, 2], [2, 0]]
# print(f"Directed graph has a cycle: {hasCycleDirected(3, edges4)}") # Expected: True


# -----------------------------------------------------------------------------
# 45. Detect Cycle in an Undirected Graph
# A DFS-based algorithm to detect a cycle in an undirected graph.
def hasCycleUndirected(num_nodes: int, edges: list[list[int]]) -> bool:
    """
    Detects if an undirected graph contains a cycle. This is a DFS-based
    approach where we track the parent of each node to distinguish a back edge
    (a cycle) from a tree edge.

    Time Complexity: O(V + E).
    Space Complexity: O(V + E) for the adjacency list and visited array.

    Args:
        num_nodes: The number of nodes.
        edges: The list of undirected edges.

    Returns:
        True if a cycle exists, False otherwise.
    """
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = [False] * num_nodes

    def dfs(node, parent):
        visited[node] = True

        for neighbor in adj[node]:
            if not visited[neighbor]:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True  # Found a back edge to an ancestor

        return False

    for i in range(num_nodes):
        if not visited[i]:
            if dfs(i, -1):
                return True

    return False


# --- Example Usage ---
# edges5 = [[0, 1], [1, 2], [2, 0], [1, 3]]
# print(f"Undirected graph has a cycle: {hasCycleUndirected(4, edges5)}") # Expected: True


# -----------------------------------------------------------------------------
# 46. Diameter of a Binary Tree
# Finds the longest path between any two nodes in a tree.
def diameterOfBinaryTree(root: TreeNode) -> int:
    """
    Calculates the diameter of a binary tree, which is the length of the
    longest path between any two nodes. This is a recursive post-order
    traversal approach.

    Time Complexity: O(N), where N is the number of nodes.
    Space Complexity: O(H) for the recursion stack, where H is the height of the tree.

    Args:
        root: The root node of the binary tree.

    Returns:
        The diameter of the tree.
    """
    max_diameter = 0

    def post_order_traversal(node):
        nonlocal max_diameter
        if not node:
            return 0

        left_height = post_order_traversal(node.left)
        right_height = post_order_traversal(node.right)

        # Update the maximum diameter found so far
        max_diameter = max(max_diameter, left_height + right_height)

        # Return the height of the current subtree
        return 1 + max(left_height, right_height)

    post_order_traversal(root)
    return max_diameter


# --- Example Usage ---
# root6 = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
# print(f"Diameter of the tree: {diameterOfBinaryTree(root6)}") # Expected: 3


# -----------------------------------------------------------------------------
# 47. Max Area of Island
# Finds the maximum size of a connected component of '1's in a grid.
def maxAreaOfIsland(grid: list[list[int]]) -> int:
    """
    Finds the maximum area of a connected component of '1's in a binary grid.
    This is a variant of the "Number of Islands" problem, using a DFS to
    calculate the size of each island.

    Time Complexity: O(R * C).
    Space Complexity: O(R * C) for the recursion stack.

    Args:
        grid: The input binary grid.

    Returns:
        The maximum area found.
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    max_area = 0

    def dfs(r, c):
        if not (0 <= r < rows and 0 <= c < cols and grid[r][c] == 1):
            return 0

        grid[r][c] = 0  # Mark as visited

        return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                max_area = max(max_area, dfs(r, c))

    return max_area


# --- Example Usage ---
# grid7 = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
# print(f"Max area of island: {maxAreaOfIsland(grid7)}") # Expected: 6


# ----------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 49. Subtree of Another Tree
# Checks if one binary tree is a subtree of another.
def isSubtree(root: TreeNode, subRoot: TreeNode) -> bool:
    """
    Checks if a binary tree 'subRoot' is a subtree of another binary tree 'root'.
    A subtree is defined as a tree where 'subRoot's root node is identical to a
    node in 'root', and their entire subtrees are identical.

    Time Complexity: O(N * M), where N and M are the number of nodes in the two trees.
                     In the worst case, we might check for a match at every node of the main tree.
    Space Complexity: O(H) for the recursion stack, where H is the height of the main tree.

    Args:
        root: The main binary tree.
        subRoot: The potential subtree.

    Returns:
        True if subRoot is a subtree of root, False otherwise.
    """

    def is_same_tree(p, q):
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

    if not root:
        return False

    if is_same_tree(root, subRoot):
        return True

    return isSubtree(root.left, subRoot) or isSubtree(root.right, subRoot)


# --- Example Usage ---
# root9 = TreeNode(3, TreeNode(4, TreeNode(1), TreeNode(2)), TreeNode(5))
# subroot9 = TreeNode(4, TreeNode(1), TreeNode(2))
# print(f"Is subroot a subtree of root? {isSubtree(root9, subroot9)}") # Expected: True


# ------------------------------------------------------------------------------------------------------


# 40. Lowest Common Ancestor (LCA) of a Binary Tree
# Finds the LCA of two nodes in a binary tree.
def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    Finds the lowest common ancestor (LCA) of two nodes in a binary tree.
    This solution uses a recursive post-order traversal. The LCA is either
    one of the nodes, or the first node where the two nodes appear in different subtrees.

    Time Complexity: O(N), where N is the number of nodes in the tree.
    Space Complexity: O(H) for the recursion stack, where H is the height of the tree.

    Args:
        root: The root of the binary tree.
        p: The first node.
        q: The second node.

    Returns:
        The lowest common ancestor node.
    """
    if root is None or root == p or root == q:
        return root

    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    if left and right:
        return root

    return left or right


# --- Example Usage ---
# root_40 = TreeNode(3, TreeNode(5, TreeNode(6), TreeNode(2, TreeNode(7), TreeNode(4))), TreeNode(1, TreeNode(0), TreeNode(8)))
# p_40, q_40 = root_40.left, root_40.right.right # Nodes 5 and 8
# # print(f"LCA of 5 and 8 is: {lowestCommonAncestor(root_40, p_40, q_40).val}") # Expected: 3


# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 42. Number of Islands
# Counts the number of distinct islands in a binary grid.
def numIslands(grid: list[list[str]]) -> int:
    """
    Counts the number of islands in a binary grid where '1' represents land
    and '0' represents water. An island is a group of '1's connected
    horizontally or vertically. This is solved by iterating through the grid and
    running a DFS or BFS from each unvisited '1' to find and mark the entire island.

    Time Complexity: O(R * C), where R and C are the grid dimensions.
    Space Complexity: O(R * C) for the recursion stack (DFS) or queue (BFS).

    Args:
        grid: The 2D grid of '1's and '0's.

    Returns:
        The total number of islands.
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    num_islands = 0

    def dfs(r, c):
        if not (0 <= r < rows and 0 <= c < cols and grid[r][c] == "1"):
            return

        grid[r][c] = "0"  # Mark as visited

        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                num_islands += 1
                dfs(r, c)

    return num_islands


# --- Example Usage ---
# grid_42 = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]
# print(f"Number of islands: {numIslands(grid_42)}") # Expected: 3


# -----------------------------------------------------------------------------
# 45. Pacific Atlantic Water Flow
# Finds cells from which water can flow to both the Pacific and Atlantic oceans.
def pacificAtlantic(heights: list[list[int]]) -> list[list[int]]:
    """
    Finds all cells in a matrix from which water can flow to both the Pacific
    (top and left edges) and Atlantic (bottom and right edges) oceans. This is
    solved by performing two separate DFS/BFS traversals, one from each "ocean"
    and then finding the intersection of the reachable cells.

    Time Complexity: O(R * C), where R and C are the matrix dimensions.
    Space Complexity: O(R * C) for the visited sets and recursion stack.

    Args:
        heights: The matrix of terrain heights.

    Returns:
        A list of [row, col] pairs of cells that can reach both oceans.
    """
    if not heights or not heights[0]:
        return []

    rows, cols = len(heights), len(heights[0])
    pacific_reachable = set()
    atlantic_reachable = set()

    def dfs(r, c, reachable_set):
        if (r, c) in reachable_set:
            return

        reachable_set.add((r, c))

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and heights[nr][nc] >= heights[r][c]:
                dfs(nr, nc, reachable_set)

    # Start DFS from Pacific borders
    for r in range(rows):
        dfs(r, 0, pacific_reachable)
    for c in range(cols):
        dfs(0, c, pacific_reachable)

    # Start DFS from Atlantic borders
    for r in range(rows):
        dfs(r, cols - 1, atlantic_reachable)
    for c in range(cols):
        dfs(rows - 1, c, atlantic_reachable)

    return list(pacific_reachable.intersection(atlantic_reachable))


# --- Example Usage ---
# heights_45 = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
# print(f"Pacific Atlantic flow cells: {pacificAtlantic(heights_45)}")
# Expected: [[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (order may vary)


# -----------------------------------------------------------------------------
# 49. All Paths From Source to Target
# Finds all paths from a source node to a target node in a directed acyclic graph.
def allPathsSourceTarget(graph: list[list[int]]) -> list[list[int]]:
    """
    Finds all possible paths from node 0 to node n-1 in a directed acyclic graph.
    This is a backtracking problem solved with Depth-First Search (DFS).

    Time Complexity: O(2^N * N), as in the worst case, the number of paths can be exponential.
    Space Complexity: O(2^N * N) to store all paths.

    Args:
        graph: An adjacency list representation of the graph.

    Returns:
        A list of all paths.
    """
    n = len(graph)
    target = n - 1
    result = []

    def dfs(node, path):
        path.append(node)

        if node == target:
            result.append(list(path))
            return

        for neighbor in graph[node]:
            dfs(neighbor, path)
            path.pop()  # Backtrack

    dfs(0, [])
    return result


# --- Example Usage ---
# graph_49 = [[1,2],[3],[3],[]]
# print(f"All paths from 0 to 3: {allPathsSourceTarget(graph_49)}") # Expected: [[0, 1, 3], [0, 2, 3]]
