[4/26, 11:20] 🤣: [4/26, 11:07] Meta AI: Here are solutions to various tree algorithms:

1. Height of a Tree
- *Problem*: Find the height of a tree.
- *Approach*: Use recursive or iterative DFS to find the maximum depth of the tree.
- *Code*:
```
def height(root):
    if not root:
        return 0
    return 1 + max(height(root.left), height(root.right))
```

2. Balanced Tree
- *Problem*: Check if a tree is balanced.
- *Approach*: Use recursive DFS to check if the height difference between left and right subtrees is within a certain threshold.
- *Code*:
```
def is_balanced(root):
    def check(node):
        if not node:
            return 0
        left_height = check(node.left)
        right_height = check(node.right)
        if left_height == -1 or right_height == -1 or abs(left_height - right_height) > 1:
            return -1
        return 1 + max(left_height, right_height)
    return check(root) != -1
```

3. Diameter of a Tree
- *Problem*: Find the diameter of a tree.
- *Approach*: Use recursive DFS to find the maximum path length between two nodes.
- *Code*:
```
def diameter(root):
    max_diameter = 0
    def dfs(node):
        nonlocal max_diameter
        if not node:
            return 0
        left_height = dfs(node.left)
        right_height = dfs(node.right)
        max_diameter = max(max_diameter, left_height + right_height)
        return 1 + max(left_height, right_height)
    dfs(root)
    return max_diameter
```

4. Sum of Tree Nodes
- *Problem*: Find the sum of all node values in a tree.
- *Approach*: Use recursive or iterative DFS to sum up node values.
- *Code*:
```
def sum_tree(root):
    if not root:
        return 0
    return root.val + sum_tree(root.left) + sum_tree(root.right)
```

5. Symmetrical Tree
- *Problem*: Check if a tree is symmetrical.
- *Approach*: Use recursive DFS to compare left and right subtrees.
- *Code*:
```
def is_symmetrical(root):
    def is_mirror(node1, node2):
        if not node1 and not node2:
            return True
        if not node1 or not node2:
            return False
        return node1.val == node2.val and is_mirror(node1.left, node2.right) and is_mirror(node1.right, node2.left)
    return is_mirror(root.left, root.right)
```

6. Identical Trees
- *Problem*: Check if two trees are identical.
- *Approach*: Use recursive DFS to compare node values and structure.
- *Code*:
```
def is_identical(root1, root2):
    if not root1 and not root2:
        return True
    if not root1 or not root2:
        return False
    return root1.val == root2.val and is_identical(root1.left, root2.left) and is_identical(root1.right, root2.right)
```

7. Zigzag Traversal
- *Problem*: Perform zigzag traversal of a tree.
- *Approach*: Use level order traversal with a flag to toggle direction.
- *Code*:
```
def zigzag_traversal(root):
    result = []
    queue = [root]
    left_to_right = True
    while queue:
        level_size = len(queue)
        level_nodes = []
        for _ in range(level_size):
            node = queue.pop(0)
            if left_to_right:
                level_nodes.append(node.val)
            else:
                level_nodes.insert(0, node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level_nodes)
        left_to_right = not left_to_right
    return result
```

8. Boundary Traversal
- *Problem*: Perform boundary traversal of a tree.
- *Approach*: Use DFS to traverse left boundary, leaves, and right boundary.
- *Code*:
```
def boundary_traversal(root):
    result = []
    def add_left_boundary(node):
        if node:
            if node.left or node.right:
                result.append(node.val)
            add_left_boundary(node.left)
    def add_leaves(node):
        if node:
            add_leaves(node.left)
            if not node.left and not node.right:
                result.append(node.val)
            add_leaves(node.right)
    def add_right_boundary(node):
        if node:
            if node.left or node.right:
                result.append(node.val)
            add_right_boundary(node.right)
    result.append(root.val)
    add_left_boundary(root.left)
    add
```
[4/26, 11:09] Meta AI: Here are solutions to various tree algorithms:

1. Vertical Order Traversal
- *Problem*: Traverse a tree in vertical order.
- *Approach*: Use a hash map to store nodes at each horizontal distance.
- *Code*:
```
def vertical_order(root):
    if not root:
        return []
    columnTable = {}
    min_column = max_column = 0
    queue = [(root, 0)]
    while queue:
        node, column = queue.pop(0)
        if node is not None:
            if column in columnTable:
                columnTable[column].append(node.val)
            else:
                columnTable[column] = [node.val]
            min_column = min(min_column, column)
            max_column = max(max_column, column)
            queue.append((node.left, column - 1))
            queue.append((node.right, column + 1))
    return [columnTable[x] for x in range(min_column, max_column + 1)]
```

2. Top View
- *Problem*: Find the top view of a tree.
- *Approach*: Use a hash map to store the first node at each horizontal distance.
- *Code*:
```
def top_view(root):
    if not root:
        return []
    columnTable = {}
    min_column = max_column = 0
    queue = [(root, 0)]
    while queue:
        node, column = queue.pop(0)
        if node is not None:
            if column not in columnTable:
                columnTable[column] = node.val
            min_column = min(min_column, column)
            max_column = max(max_column, column)
            queue.append((node.left, column - 1))
            queue.append((node.right, column + 1))
    return [columnTable[x] for x in range(min_column, max_column + 1)]
```

3. Bottom View
- *Problem*: Find the bottom view of a tree.
- *Approach*: Use a hash map to store the last node at each horizontal distance.
- *Code*:
```
def bottom_view(root):
    if not root:
        return []
    columnTable = {}
    min_column = max_column = 0
    queue = [(root, 0)]
    while queue:
        node, column = queue.pop(0)
        if node is not None:
            columnTable[column] = node.val
            min_column = min(min_column, column)
            max_column = max(max_column, column)
            queue.append((node.left, column - 1))
            queue.append((node.right, column + 1))
    return [columnTable[x] for x in range(min_column, max_column + 1)]
```

4. Left View
- *Problem*: Find the left view of a tree.
- *Approach*: Use level order traversal and store the first node at each level.
- *Code*:
```
def left_view(root):
    if not root:
        return []
    result = []
    queue = [root]
    while queue:
        level_size = len(queue)
        result.append(queue[0].val)
        for _ in range(level_size):
            node = queue.pop(0)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return result
```

5. Right View
- *Problem*: Find the right view of a tree.
- *Approach*: Use level order traversal and store the last node at each level.
- *Code*:
```
def right_view(root):
    if not root:
        return []
    result = []
    queue = [root]
    while queue:
        level_size = len(queue)
        result.append(queue[-1].val)
        for _ in range(level_size):
            node = queue.pop(0)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return result
```

6. Lowest Common Ancestor
- *Problem*: Find the lowest common ancestor of two nodes.
- *Approach*: Use recursive DFS to find the LCA.
- *Code*:
```
def lowest_common_ancestor(root, p, q):
    if not root:
        return None
    if root == p or root == q:
        return root
    left_lca = lowest_common_ancestor( root.left, p, q)
    right_lca = lowest_common_ancestor( root.right, p, q)
    if left_lca and right_lca:
        return root
    return left_lca if left_lca else right_lca
```

7. Time Taken to Burn a Tree
- *Problem*: Find the time taken to burn a tree from a given node.
- *Approach*: Use BFS to find the maximum distance from the given node.
- *Code*:
```
def time_to_burn(root
```
[4/26, 11:11] Meta AI: Here are solutions to various tree algorithms:

1. Width of a Tree
- *Problem*: Find the maximum width of a tree.
- *Approach*: Use level order traversal to find the maximum number of nodes at any level.
- *Code*:
```
def width(root):
    if not root:
        return 0
    max_width = 0
    queue = [root]
    while queue:
        level_size = len(queue)
        max_width = max(max_width, level_size)
        for _ in range(level_size):
            node = queue.pop(0)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return max_width
```

2. Sum of Tree Nodes
- *Problem*: Find the sum of all node values in a tree.
- *Approach*: Use recursive or iterative DFS to sum up node values.
- *Code*:
```
def sum_tree(root):
    if not root:
        return 0
    return root.val + sum_tree(root.left) + sum_tree(root.right)
```

3. Nodes at Distance K
- *Problem*: Find all nodes at a distance K from a given node.
- *Approach*: Use DFS or BFS to find nodes at distance K.
- *Code*:
```
def nodes_at_distance_k(root, target, k):
    if not root:
        return []
    result = []
    def dfs(node, distance):
        if not node:
            return
        if distance == k:
            result.append(node.val)
        dfs(node.left, distance + 1)
        dfs(node.right, distance + 1)
    dfs(target, 0)
    return result
```

4. Time Taken to Burn a Tree
- *Problem*: Find the time taken to burn a tree from a given node.
- *Approach*: Use BFS to find the maximum distance from the given node.
- *Code*:
```
def time_to_burn(root, target):
    if not root:
        return 0
    parent_map = {}
    def build_parent_map(node, parent):
        if node:
            parent_map[node] = parent
            build_parent_map(node.left, node)
            build_parent_map(node.right, node)
    build_parent_map(root, None)
    queue = [target]
    visited = set([target])
    time = 0
    while queue:
        level_size = len(queue)
        for _ in range(level_size):
            node = queue.pop(0)
            if node.left and node.left not in visited:
                queue.append(node.left)
                visited.add(node.left)
            if node.right and node.right not in visited:
                queue.append(node.right)
                visited.add(node.right)
            if parent_map[node] and parent_map[node] not in visited:
                queue.append(parent_map[node])
                visited.add(parent_map[node])
        time += 1
    return time - 1
```

5. Complete Tree Node Count
- *Problem*: Count the number of nodes in a complete binary tree.
- *Approach*: Use recursive or iterative approach to count nodes.
- *Code*:
```
def count_nodes(root):
    if not root:
        return 0
    return 1 + count_nodes(root.left) + count_nodes(root.right)
```

6. Unique Binary Trees
- *Problem*: Find the number of unique binary trees that can be formed with n nodes.
- *Approach*: Use dynamic programming to calculate the number of unique trees.
- *Code*:
```
def num_unique_trees(n):
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - j - 1]
    return dp[n]
```

7. Construction using Preorder
- *Problem*: Construct a binary tree from preorder traversal.
- *Approach*: Use recursive approach to construct the tree.
- *Code*:
```
def construct_from_preorder(preorder):
    if not preorder:
        return None
    root = TreeNode(preorder[0])
    stack = [root]
    for value in preorder[1:]:
        if value < stack[-1].val:
            stack[-1].left = TreeNode(value)
            stack.append(stack[-1].left)
        else:
            while stack and stack[-1].val < value:
                last_node = stack.pop()
            last_node.right = TreeNode(value)
            stack.append(last_node.right)
    return root
```

8. Construction using Postorder
- *Problem*: Construct a binary tree from postorder traversal.
- *Approach*: Use recursive approach to construct the tree.
- *Code*:
```
def construct_from
```
[4/26, 11:12] Meta AI: Here are solutions to tree algorithms construction using inorder:

1. Construction using Inorder and Preorder
- *Problem*: Construct a binary tree from inorder and preorder traversals.
- *Approach*: Use recursive approach to construct the tree.
- *Code*:
```
def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root = TreeNode(preorder[0])
    index = inorder.index(preorder[0])
    root.left = build_tree(preorder[1:index + 1], inorder[:index])
    root.right = build_tree(preorder[index + 1:], inorder[index + 1:])
    return root
```

2. Construction using Inorder and Postorder
- *Problem*: Construct a binary tree from inorder and postorder traversals.
- *Approach*: Use recursive approach to construct the tree.
- *Code*:
```
def build_tree(inorder, postorder):
    if not inorder or not postorder:
        return None
    root = TreeNode(postorder[-1])
    index = inorder.index(postorder[-1])
    root.left = build_tree(inorder[:index], postorder[:index])
    root.right = build_tree(inorder[index + 1:], postorder[index:-1])
    return root
```

These algorithms work by recursively constructing the left and right subtrees based on the inorder traversal, which provides the order of nodes in the tree. The preorder or postorder traversal provides the root node, which is used to divide the inorder traversal into left and right subtrees.

Example Usage
```
Example usage for construction using inorder and preorder
preorder = [3, 9, 20, 15, 7]
inorder = [9, 3, 15, 20, 7]
root = build_tree(preorder, inorder)

Example usage for construction using inorder and postorder
inorder = [9, 3, 15, 20, 7]
postorder = [9, 15, 7, 20, 3]
root = build_tree(inorder, postorder)
```
[4/26, 16:15] 🤣: [4/26, 15:57] Meta AI: Here are solutions to various graph algorithms:

1. Breadth-First Search (BFS)
- *Problem*: Traverse a graph level by level, starting from a given node.
- *Approach*: Use a queue to keep track of nodes to visit.
- *Code*:
```
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
```

2. Depth-First Search (DFS)
- *Problem*: Traverse a graph by visiting a node and then visiting all of its neighbors before backtracking.
- *Approach*: Use a stack or recursion to keep track of nodes to visit.
- *Code*:
```
def dfs(graph, start):
    visited = set()
    def recursive_dfs(node):
        visited.add(node)
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                recursive_dfs(neighbor)
    recursive_dfs(start)
```

3. Number of Provinces
- *Problem*: Find the number of connected components in a graph.
- *Approach*: Use DFS or BFS to traverse the graph and count the number of connected components.
- *Code*:
```
def find_provinces(graph):
    visited = set()
    count = 0
    for node in graph:
        if node not in visited:
            dfs(graph, node, visited)
            count += 1
    return count
```

4. Number of Islands
- *Problem*: Find the number of islands in a grid.
- *Approach*: Use DFS or BFS to traverse the grid and count the number of islands.
- *Code*:
```
def num_islands(grid):
    if not grid:
        return 0
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(grid, i, j)
                count += 1
    return count
```

5. Flood Fill
- *Problem*: Fill a region of a grid with a new color.
- *Approach*: Use DFS or BFS to traverse the region and fill it with the new color.
- *Code*:
```
def flood_fill(grid, x, y, new_color):
    old_color = grid[x][y]
    if old_color == new_color:
        return grid
    def dfs(i, j):
        if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == old_color:
            grid[i][j] = new_color
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)
    dfs(x, y)
    return grid
```

6. Rotten Orange
- *Problem*: Find the minimum time required to rot all oranges in a grid.
- *Approach*: Use BFS to traverse the grid and find the minimum time.
- *Code*:
```
from collections import deque

def oranges_rotting(grid):
    queue = deque()
    fresh_count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 2:
                queue.append((i, j))
            elif grid[i][j] == 1:
                fresh_count += 1
    minutes = 0
    while queue and fresh_count:
        minutes += 1
        for _ in range(len(queue)):
            x, y = queue.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 1:
                    grid[nx][ny] = 2
                    queue.append((nx, ny))
                    fresh_count -= 1
    return minutes if fresh_count == 0 else -1
```

7. Detect Cycle in Undirected Graph
- *Problem*: Detect whether a cycle exists in an undirected graph.
- *Approach*: Use DFS to traverse the graph and detect cycles.
- *Code*:
```
def has_cycle(graph):
    visited = set()
    for node in graph:
        if node not in visited:
            if dfs(graph, node, visited, None):
                return True
    return False

def dfs(graph, node, visited, parent):
    visited.add(node)
```
[4/26, 15:59] Meta AI: Here are solutions to various graph algorithms:

1. Detect Cycle in Undirected Graph
- *Problem*: Detect whether a cycle exists in an undirected graph.
- *Approach*: Use DFS to traverse the graph and detect cycles.
- *Code*:
```
def has_cycle(graph):
    visited = set()
    for node in graph:
        if node not in visited:
            if dfs(graph, node, visited, None):
                return True
    return False

def dfs(graph, node, visited, parent):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            if dfs(graph, neighbor, visited, node):
                return True
        elif neighbor != parent:
            return True
    return False
```

2. Detect Cycle in Directed Graph
- *Problem*: Detect whether a cycle exists in a directed graph.
- *Approach*: Use DFS to traverse the graph and detect cycles.
- *Code*:
```
def has_cycle(graph):
    visited = set()
    recursion_stack = set()
    for node in graph:
        if node not in visited:
            if dfs(graph, node, visited, recursion_stack):
                return True
    return False

def dfs(graph, node, visited, recursion_stack):
    visited.add(node)
    recursion_stack.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            if dfs(graph, neighbor, visited, recursion_stack):
                return True
        elif neighbor in recursion_stack:
            return True
    recursion_stack.remove(node)
    return False
```

3. Distance Nearest
- *Problem*: Find the shortest distance from a given node to all other nodes in a graph.
- *Approach*: Use BFS to traverse the graph and calculate distances.
- *Code*:
```
from collections import deque

def distance_nearest(graph, start):
    queue = deque([(start, 0)])
    distances = {start: 0}
    while queue:
        node, distance = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in distances or distances[neighbor] > distance + 1:
                distances[neighbor] = distance + 1
                queue.append((neighbor, distance + 1))
    return distances
```

4. Surrounded Regions
- *Problem*: Capture all regions surrounded by 'X' in a grid.
- *Approach*: Use DFS to traverse the grid and capture surrounded regions.
- *Code*:
```
def solve(board):
    if not board:
        return
    m, n = len(board), len(board[0])
    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n and board[i][j] == 'O':
            board[i][j] = 'N'
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)
    for i in range(m):
        dfs(i, 0)
        dfs(i, n - 1)
    for j in range(n):
        dfs(0, j)
        dfs(m - 1, j)
    for i in range(m):
        for j in range(n):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == 'N':
                board[i][j] = 'O'
```

5. Number of Enclaves
- *Problem*: Find the number of 'O' regions that are not connected to the boundary.
- *Approach*: Use DFS to traverse the grid and count enclaves.
- *Code*:
```
def num_enclaves(grid):
    m, n = len(grid), len(grid[0])
    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n and grid[i][j] == 1:
            grid[i][j] = 0
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)
    for i in range(m):
        dfs(i, 0)
        dfs(i, n - 1)
    for j in range(n):
        dfs(0, j)
        dfs(m - 1, j)
    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                count += 1
    return count
```

6. Distinct Islands
- *Problem*: Find the number of distinct islands in a grid.
- *Approach*: Use DFS to traverse the grid and count distinct islands.
- *Code*:
```
def num_distinct_islands(grid):
    m, n = len(grid), len
```
[4/26, 16:01] Meta AI: Here are solutions to various graph algorithms:

1. Bipartite Graph
- *Problem*: Determine if a graph is bipartite.
- *Approach*: Use DFS or BFS to traverse the graph and assign colors to nodes.
- *Code*:
```
def is_bipartite(graph):
    colors = {}
    for node in graph:
        if node not in colors:
            if not dfs(graph, node, colors, 0):
                return False
    return True

def dfs(graph, node, colors, color):
    colors[node] = color
    for neighbor in graph[node]:
        if neighbor not in colors:
            if not dfs(graph, neighbor, colors, 1 - color):
                return False
        elif colors[neighbor] == color:
            return False
    return True
```

2. Find Eventual Safe State
- *Problem*: Find all nodes that are eventual safe states in a directed graph.
- *Approach*: Use DFS to traverse the graph and identify safe nodes.
- *Code*:
```
def eventual_safe_nodes(graph):
    safe = [False] * len(graph)
    visited = [False] * len(graph)
    def dfs(node):
        if visited[node]:
            return safe[node]
        visited[node] = True
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        safe[node] = True
        return True
    for i in range(len(graph)):
        dfs(i)
    return [i for i in range(len(graph)) if safe[i]]
```

3. Topological Sort
- *Problem*: Perform a topological sort on a directed acyclic graph (DAG).
- *Approach*: Use DFS or Kahn's algorithm to traverse the graph and order nodes.
- *Code*:
```
def topological_sort(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    queue = [node for node in graph if in_degree[node] == 0]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return result
```

4. Kahn's Algorithm
- *Problem*: Perform a topological sort using Kahn's algorithm.
- *Approach*: Use a queue to traverse the graph and order nodes.
- *Code*:
```
def kahns_algorithm(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    queue = [node for node in graph if in_degree[node] == 0]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return result
```

5. Course Schedule
- *Problem*: Determine if a course schedule is valid.
- *Approach*: Use DFS or topological sort to traverse the graph and detect cycles.
- *Code*:
```
def can_finish(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    visited = [0] * num_courses
    for course, prerequisite in prerequisites:
        graph[prerequisite].append(course)
    def dfs(node):
        if visited[node] == -1:
            return False
        if visited[node] == 1:
            return True
        visited[node] = -1
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        visited[node] = 1
        return True
    for i in range(num_courses):
        if not dfs(i):
            return False
    return True
```

6. Shortest Path in Directed Graph
- *Problem*: Find the shortest path from a source node to all other nodes in a directed graph.
- *Approach*: Use Dijkstra's algorithm or Bellman-Ford algorithm to traverse the graph and calculate distances.
- *Code*:
```
import heapq

def shortest_path(graph, source):
    distances = {node: float('inf') for node in graph}
    distances[source] = 0
    queue = [(0, source)]
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor
```
[4/26, 16:04] Meta AI: Here are solutions to various graph algorithms:

1. Eventual Safe State
- *Problem*: Find all nodes that are eventual safe states in a directed graph.
- *Approach*: Use DFS to traverse the graph and identify safe nodes.
- *Code*:
```
def eventual_safe_nodes(graph):
    n = len(graph)
    safe = [False] * n
    visited = [False] * n
    def dfs(node):
        if visited[node]:
            return safe[node]
        visited[node] = True
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        safe[node] = True
        return True
    for i in range(n):
        dfs(i)
    return [i for i in range(n) if safe[i]]
```

2. Shortest Path in Directed Graph
- *Problem*: Find the shortest path from a source node to all other nodes in a directed graph.
- *Approach*: Use Dijkstra's algorithm to traverse the graph and calculate distances.
- *Code*:
```
import heapq

def shortest_path(graph, source):
    distances = {node: float('inf') for node in graph}
    distances[source] = 0
    queue = [(0, source)]
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    return distances
```

3. Shortest Path in Undirected Graph
- *Problem*: Find the shortest path from a source node to all other nodes in an undirected graph.
- *Approach*: Use Dijkstra's algorithm or BFS to traverse the graph and calculate distances.
- *Code*:
```
import heapq

def shortest_path(graph, source):
    distances = {node: float('inf') for node in graph}
    distances[source] = 0
    queue = [(0, source)]
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor in graph[current_node]:
            distance = current_distance + 1
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    return distances
```

4. Word Ladder 1
- *Problem*: Find the shortest transformation sequence from a begin word to an end word.
- *Approach*: Use BFS to traverse the graph and find the shortest path.
- *Code*:
```
from collections import deque

def ladder_length(begin_word, end_word, word_list):
    word_set = set(word_list)
    queue = deque([(begin_word, 1)])
    while queue:
        word, length = queue.popleft()
        if word == end_word:
            return length
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i + 1:]
                if next_word in word_set:
                    word_set.remove(next_word)
                    queue.append((next_word, length + 1))
    return 0
```

5. Dijkstra's Algorithm using Set
- *Problem*: Implement Dijkstra's algorithm using a set data structure.
- *Approach*: Use a set to keep track of nodes with minimum distance.
- *Code*:
```
import heapq

def dijkstra(graph, source):
    distances = {node: float('inf') for node in graph}
    distances[source] = 0
    queue = [(0, source)]
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    return distances
```

6. Path with Minimum Effort
- *Problem*: Find the path with minimum effort in a graph.
- *Approach*: Use Dijkstra's algorithm to traverse the graph and calculate distances.
- *Code*:
```
import heapq

def minimum_effort_path(grid):
    m, n = len(grid), len(grid[0])
    efforts = [[float('inf')] * n for _ in range(m)]
    efforts[0][0] = 0
    queue = [(0, 0, 0)]
    while queue:
        effort, x, y = heapq.heappop(queue)
        if effort > efforts[x][y]:
            continue
        for dx, dy
```
[4/26, 16:10] Meta AI: Here are solutions to various graph algorithms:

1. Cheapest Flight with K Stops
- *Problem*: Find the cheapest flight from a source city to a destination city with at most K stops.
- *Approach*: Use Bellman-Ford algorithm or Dijkstra's algorithm with a priority queue to traverse the graph and calculate distances.
- *Code*:
```
import heapq

def find_cheapest_price(n, flights, src, dst, k):
    prices = [float('inf')] * n
    prices[src] = 0
    for _ in range(k + 1):
        temp_prices = prices[:]
        for u, v, price in flights:
            if prices[u] != float('inf') and prices[u] + price < temp_prices[v]:
                temp_prices[v] = prices[u] + price
        prices = temp_prices
    return prices[dst] if prices[dst] != float('inf') else -1
```

2. Minimum Multiplication to Reach End
- *Problem*: Find the minimum number of multiplications required to reach the end node from the start node in a graph.
- *Approach*: Use BFS or Dijkstra's algorithm to traverse the graph and calculate distances.
- *Code*:
```
import heapq

def min_operations(nums, start, end):
    queue = [(0, start)]
    visited = set()
    while queue:
        operations, current = heapq.heappop(queue)
        if current == end:
            return operations
        if current in visited:
            continue
        visited.add(current)
        for num in nums:
            next_num = current * num
            if next_num not in visited:
                heapq.heappush(queue, (operations + 1, next_num))
    return -1
```

3. Number of Ways to Arrive at Destination
- *Problem*: Find the number of ways to arrive at a destination node from a source node in a graph.
- *Approach*: Use dynamic programming to traverse the graph and calculate the number of ways.
- *Code*:
```
def count_ways(graph, source, destination):
    ways = [0] * len(graph)
    ways[source] = 1
    for _ in range(len(graph)):
        for u in range(len(graph)):
            for v in graph[u]:
                ways[v] += ways[u]
    return ways[destination]
```

4. Bellman-Ford Algorithm
- *Problem*: Find the shortest path from a source node to all other nodes in a weighted graph.
- *Approach*: Use Bellman-Ford algorithm to traverse the graph and calculate distances.
- *Code*:
```
def bellman_ford(graph, source):
    distances = [float('inf')] * len(graph)
    distances[source] = 0
    for _ in range(len(graph) - 1):
        for u in range(len(graph)):
            for v, weight in graph[u]:
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
    for u in range(len(graph)):
        for v, weight in graph[u]:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                return None  # negative cycle detected
    return distances
```

5. Floyd-Warshall Algorithm
- *Problem*: Find the shortest path between all pairs of nodes in a weighted graph.
- *Approach*: Use Floyd-Warshall algorithm to traverse the graph and calculate distances.
- *Code*:
```
def floyd_warshall(graph):
    n = len(graph)
    distances = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        distances[i][i] = 0
    for u in range(n):
        for v, weight in graph[u]:
            distances[u][v] = weight
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])
    return distances
```

6. Prim's Algorithm
- *Problem*: Find the minimum spanning tree of a graph.
- *Approach*: Use Prim's algorithm to traverse the graph and calculate the minimum spanning tree.
- *Code*:
```
import heapq

def prim(graph):
    mst = []
    visited = set()
    edges = [(0, None, 0)]  # (weight, parent, node)
    while edges:
        weight, parent, node = heapq.heappop(edges)
        if node in visited:
            continue
        visited.add(node)
        if parent is not None:
            mst.append((parent, node, weight))
        for neighbor, neighbor_weight in
```
[4/26, 16:11] Meta AI: Here's an explanation of Prim's algorithm:

What is Prim's Algorithm?
Prim's algorithm is a greedy algorithm used to find the minimum spanning tree (MST) of a connected, undirected, and weighted graph.

How Does Prim's Algorithm Work?
1. *Initialization*: Choose an arbitrary node as the starting node.
2. *Grow the Tree*: At each step, select the minimum-weight edge that connects a node in the tree to a node not yet in the tree.
3. *Add the Edge*: Add the selected edge to the tree.
4. *Repeat*: Repeat steps 2-3 until all nodes are included in the tree.

Example Code
```
import heapq

def prim(graph):
    mst = []
    visited = set()
    edges = [(0, None, 0)]  # (weight, parent, node)
    while edges:
        weight, parent, node = heapq.heappop(edges)
        if node in visited:
            continue
        visited.add(node)
        if parent is not None:
            mst.append((parent, node, weight))
        for neighbor, neighbor_weight in graph[node].items():
            if neighbor not in visited:
                heapq.heappush(edges, (neighbor_weight, node, neighbor))
    return mst

Example usage:
graph = {
    0: {1: 2, 3: 6},
    1: {0: 2, 2: 3, 3: 8, 4: 5},
    2: {1: 3, 4: 7},
    3: {0: 6, 1: 8, 4: 9},
    4: {1: 5, 2: 7, 3: 9}
}

mst = prim(graph)
print(mst)
```

Time Complexity
The time complexity of Prim's algorithm is O(E log V), where E is the number of edges and V is the number of vertices.

Space Complexity
The space complexity of Prim's algorithm is O(V + E), where V is the number of vertices and E is the number of edges.
[4/26, 22:23] 🤣: [4/26, 17:44] Meta AI: Here are solutions to various stack and queue problems:

1. Check for Balanced Parentheses
- *Problem*: Check if a string of parentheses is balanced.
- *Approach*: Use a stack to keep track of opening parentheses.
- *Code*:
```
def is_balanced(s):
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping.values():
            stack.append(char)
        elif char in mapping.keys():
            if not stack or stack.pop() != mapping[char]:
                return False
    return not stack
```

2. Prefix to Postfix Conversion
- *Problem*: Convert a prefix expression to postfix.
- *Approach*: Use a stack to keep track of operators.
- *Code*:
```
def prefix_to_postfix(prefix):
    stack = []
    operators = set(['+', '-', '*', '/'])
    for char in reversed(prefix):
        if char in operators:
            operand1 = stack.pop()
            operand2 = stack.pop()
            stack.append(operand1 + operand2 + char)
        else:
            stack.append(char)
    return stack[0]
```

3. Infix to Postfix Conversion
- *Problem*: Convert an infix expression to postfix.
- *Approach*: Use a stack to keep track of operators.
- *Code*:
```
def infix_to_postfix(infix):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    stack = []
    postfix = ''
    for char in infix:
        if char.isalpha():
            postfix += char
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                postfix += stack.pop()
            stack.pop()
        else:
            while stack and stack[-1] != '(' and precedence[char] <= precedence[stack[-1]]:
                postfix += stack.pop()
            stack.append(char)
    while stack:
        postfix += stack.pop()
    return postfix
```

4. Implement Min Stack
- *Problem*: Implement a stack that supports min operation.
- *Approach*: Use two stacks, one for elements and one for minimum elements.
- *Code*:
```
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, x):
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self):
        if self.stack:
            if self.stack[-1] == self.min_stack[-1]:
                self.min_stack.pop()
            return self.stack.pop()

    def top(self):
        if self.stack:
            return self.stack[-1]

    def getMin(self):
        if self.min_stack:
            return self.min_stack[-1]
```

5. Next Greater Element
- *Problem*: Find the next greater element for each element in an array.
- *Approach*: Use a stack to keep track of elements.
- *Code*:
```
def next_greater_element(nums):
    stack = []
    result = [-1] * len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    return result
```

6. Trapping Rain Water
- *Problem*: Calculate the amount of rain water that can be trapped.
- *Approach*: Use two pointers to keep track of maximum height.
- *Code*:
```
def trap(height):
    left, right = 0, len(height) - 1
    max_left, max_right = 0, 0
    result = 0
    while left <= right:
        if height[left] < height[right]:
            if height[left] >= max_left:
                max_left = height[left]
            else:
                result += max_left - height[left]
            left += 1
        else:
            if height[right] >= max_right:
                max_right = height[right]
            else:
                result += max_right - height[right]
            right -= 1
    return result
```

7. Largest Rectangle in Histogram
- *Problem*: Find the area of the largest rectangle in a histogram.
- *Approach*: Use a stack to keep track of indices.
- *Code*:
```
def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    for i in range(len(heights)):
        while stack and heights[stack[-1]] > heights[i]:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i
```
[4/26, 17:47] Meta AI: Here are solutions to various stack and queue problems:

1. Previous Smaller Element
- *Problem*: Find the previous smaller element for each element in an array.
- *Approach*: Use a stack to keep track of elements.
- *Code*:
```
def previous_smaller_element(nums):
    stack = []
    result = [-1] * len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]] >= nums[i]:
            stack.pop()
        if stack:
            result[i] = nums[stack[-1]]
        stack.append(i)
    return result
```

2. Sum of Subarray Minimum
- *Problem*: Calculate the sum of minimum elements of all subarrays.
- *Approach*: Use a stack to keep track of elements.
- *Code*:
```
def sum_subarray_min(nums):
    stack = []
    result = 0
    nums = [0] + nums + [0]
    for i in range(len(nums)):
        while stack and nums[stack[-1]] > nums[i]:
            mid = stack.pop()
            left = stack[-1]
            right = i
            result += nums[mid] * (mid - left) * (right - mid)
        stack.append(i)
    return result
```

3. Asteroid Collision
- *Problem*: Determine the outcome of asteroid collisions.
- *Approach*: Use a stack to keep track of asteroids.
- *Code*:
```
def asteroid_collision(asteroids):
    stack = []
    for asteroid in asteroids:
        while stack and asteroid < 0 and stack[-1] > 0:
            if stack[-1] < -asteroid:
                stack.pop()
                continue
            elif stack[-1] == -asteroid:
                stack.pop()
            break
        else:
            stack.append(asteroid)
    return stack
```

4. Largest Rectangle in Histogram
- *Problem*: Find the area of the largest rectangle in a histogram.
- *Approach*: Use a stack to keep track of indices.
- *Code*:
```
def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    for i in range(len(heights)):
        while stack and heights[stack[-1]] > heights[i]:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)
    while stack:
        h = heights[stack.pop()]
        w = len(heights) if not stack else len(heights) - stack[-1] - 1
        max_area = max(max_area, h * w)
    return max_area
```

5. Remove K Digits
- *Problem*: Remove k digits from a number to form the smallest possible number.
- *Approach*: Use a stack to keep track of digits.
- *Code*:
```
def remove_k_digits(num, k):
    stack = []
    for digit in num:
        while k and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    while k:
        stack.pop()
        k -= 1
    return ''.join(stack).lstrip('0') or '0'
```

6. Stock Span
- *Problem*: Calculate the stock span for each day.
- *Approach*: Use a stack to keep track of indices.
- *Code*:
```
def stock_span(prices):
    stack = []
    result = []
    for i, price in enumerate(prices):
        while stack and prices[stack[-1]] <= price:
            stack.pop()
        result.append(i - stack[-1] - 1 if stack else i + 1)
        stack.append(i)
    return result
```

7. Sliding Window Minimum
- *Problem*: Find the minimum element in a sliding window.
- *Approach*: Use a deque to keep track of indices.
- *Code*:
```
from collections import deque

def sliding_window_min(nums, k):
    dq = deque()
    result = []
    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] > nums[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
```

8. Celebrity Problem
- *Problem*: Find the celebrity in a group of people.
- *Approach*: Use a stack to keep track of candidates.
- *Code*:
```
def find_celebrity(matrix):
    stack = list(range(len(matrix)))
    while len(stack) > 1:
        i = stack.pop()
```
[4/26, 17:48] Meta AI: Here are solutions to LRU Cache problems:

What is LRU Cache?
LRU (Least Recently Used) Cache is a data structure that stores a limited number of items. When the cache is full and a new item needs to be added, the least recently used item is removed.

Implementing LRU Cache
- *Approach*: Use a dictionary and a doubly linked list to implement the LRU Cache.
- *Code*:
```
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return -1

    def put(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
```

Using Python's Built-in OrderedDict
- *Approach*: Use Python's built-in OrderedDict to implement the LRU Cache.
- *Code*:
```
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

Time Complexity
- *Get*: O(1)
- *Put*: O(1)

Space Complexity
- *Cache*: O(capacity)
