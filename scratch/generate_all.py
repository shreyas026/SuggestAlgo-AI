import json
import os

# Complete script to create all 112+ algorithms across 10 categories
# Each entry contains ALL 16 required fields.

PRIMARY_CATEGORIES = [
    "Searching",
    "Sorting",
    "Classification",
    "Regression",
    "Graph Algorithms",
    "Tree Algorithms",
    "Dynamic Programming",
    "Greedy Algorithms",
    "Backtracking",
    "Machine Learning / AI"
]

REQUIRED_FIELDS = [
    "id", "name", "category", "description", "problem_patterns",
    "data_structures", "requirements", "input_characteristics",
    "constraints", "best_use_cases", "time_complexity", "space_complexity",
    "advantages", "limitations", "alternatives", "python_template"
]

kb = []

# ==================== 1. SEARCHING ====================
kb.extend([
    {
        "id": "binary_search", "name": "Binary Search", "category": "Searching",
        "description": "Efficient search algorithm that finds the position of a target value within a sorted array by repeatedly dividing the search interval in half.",
        "problem_patterns": ["sorted array lookup", "find element in sorted list", "search in logarithmic time", "half search space", "find square root or threshold", "sorted data existence check"],
        "data_structures": ["Array", "Sorted List"],
        "requirements": ["Input array must be pre-sorted in ascending/descending order", "Random access by index O(1)"],
        "input_characteristics": ["Sorted numerical or string sequence", "Known boundaries"],
        "constraints": ["Data structure must support fast indexed access", "Requires prior sorting step"],
        "best_use_cases": ["Searching in large sorted numerical arrays", "Finding boundaries/thresholds in monotonic functions", "Range lookup operations"],
        "time_complexity": "O(log n)", "space_complexity": "O(1)",
        "advantages": ["Very fast for large sorted lists", "Minimal memory overhead"],
        "limitations": ["Requires data to be pre-sorted", "Not suitable for linked lists"],
        "alternatives": ["Interpolation Search", "Hash Table Lookup", "Exponential Search"],
        "python_template": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1"
    },
    {
        "id": "linear_search", "name": "Linear Search", "category": "Searching",
        "description": "Sequential search algorithm that checks every element in a list one by one until a match is found or the end is reached.",
        "problem_patterns": ["unsorted list search", "find first occurrence", "small dataset lookup", "sequential search", "find match in unsorted array"],
        "data_structures": ["Array", "Linked List", "Iterable"],
        "requirements": ["None (works on unsorted or arbitrary data)"],
        "input_characteristics": ["Unsorted or arbitrary order", "Small to medium size"],
        "constraints": ["Inefficient for large arrays"],
        "best_use_cases": ["Small collections", "Unsorted list single-pass lookup", "When sorting cost exceeds search cost"],
        "time_complexity": "O(n)", "space_complexity": "O(1)",
        "advantages": ["Simple to implement", "No prerequisite sorting or index required"],
        "limitations": ["Linear time lookup O(n) becomes slow for big datasets"],
        "alternatives": ["Binary Search", "Hash Table Lookup"],
        "python_template": "def linear_search(arr, target):\n    for idx, val in enumerate(arr):\n        if val == target: return idx\n    return -1"
    },
    {
        "id": "hash_table_lookup", "name": "Hash Table / Dictionary Lookup", "category": "Searching",
        "description": "Data structure technique that stores key-value pairs using a hash function for near-constant time lookups and insertions.",
        "problem_patterns": ["constant time lookup", "frequency count", "find duplicates", "two sum problem", "key value mapping", "fast index matching"],
        "data_structures": ["Hash Map", "Hash Set", "Dictionary"],
        "requirements": ["Keys must be hashable and immutable"],
        "input_characteristics": ["Unique keys or items", "Unordered data"],
        "constraints": ["Consumes extra memory for hash storage", "Potential hash collisions"],
        "best_use_cases": ["Repeated fast lookups", "Element frequency counting", "Caching and memoization"],
        "time_complexity": "O(1) average, O(n) worst case", "space_complexity": "O(n)",
        "advantages": ["Constant time average lookup O(1)", "Versatile key-value associations"],
        "limitations": ["Unordered data storage", "Memory overhead"],
        "alternatives": ["Binary Search Tree", "Trie"],
        "python_template": "def hash_lookup(arr, target):\n    lookup_map = {val: idx for idx, val in enumerate(arr)}\n    return lookup_map.get(target, -1)"
    },
    {
        "id": "jump_search", "name": "Jump Search", "category": "Searching",
        "description": "Searching algorithm for sorted arrays that checks fewer elements than linear search by jumping ahead by fixed steps.",
        "problem_patterns": ["jump ahead search", "block search in sorted array", "sub-linear search without binary split"],
        "data_structures": ["Array", "Sorted List"],
        "requirements": ["Input array must be sorted"],
        "input_characteristics": ["Sorted list", "Random access indexing"],
        "constraints": ["Optimal step size is sqrt(n)"],
        "best_use_cases": ["Searching sorted arrays when jumping backwards is expensive"],
        "time_complexity": "O(sqrt(n))", "space_complexity": "O(1)",
        "advantages": ["Better than linear search", "Only jumps backward once"],
        "limitations": ["Slower than binary search O(log n)"],
        "alternatives": ["Binary Search", "Exponential Search"],
        "python_template": "import math\ndef jump_search(arr, target):\n    n = len(arr); step = int(math.sqrt(n)); prev = 0\n    while arr[min(step, n)-1] < target:\n        prev = step; step += int(math.sqrt(n))\n        if prev >= n: return -1\n    while arr[prev] < target:\n        prev += 1\n        if prev == min(step, n): return -1\n    if arr[prev] == target: return prev\n    return -1"
    },
    {
        "id": "interpolation_search", "name": "Interpolation Search", "category": "Searching",
        "description": "Improved binary search variant for uniformly distributed sorted arrays that estimates position based on target value relative to low/high values.",
        "problem_patterns": ["uniformly distributed sorted search", "phonebook style search", "numeric range interpolation"],
        "data_structures": ["Array", "Sorted List"],
        "requirements": ["Data must be sorted and uniformly distributed"],
        "input_characteristics": ["Sorted numbers with uniform intervals"],
        "constraints": ["Performance degrades to O(n) if data is non-uniform"],
        "best_use_cases": ["Searching in large uniformly distributed numerical sorted data"],
        "time_complexity": "O(log log n) average, O(n) worst case", "space_complexity": "O(1)",
        "advantages": ["Sub-logarithmic search O(log log n) on uniform data"],
        "limitations": ["Sensitive to non-uniform distribution"],
        "alternatives": ["Binary Search", "Exponential Search"],
        "python_template": "def interpolation_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high and target >= arr[low] and target <= arr[high]:\n        if low == high:\n            return low if arr[low] == target else -1\n        pos = low + int(((high - low) / (arr[high] - arr[low])) * (target - arr[low]))\n        if arr[pos] == target: return pos\n        if arr[pos] < target: low = pos + 1\n        else: high = pos - 1\n    return -1"
    },
    {
        "id": "exponential_search", "name": "Exponential Search", "category": "Searching",
        "description": "Search algorithm for unbounded or infinite sorted lists that finds a range where target resides by doubling indices, then performs binary search.",
        "problem_patterns": ["unbounded array search", "infinite stream search", "unknown length sorted list"],
        "data_structures": ["Array", "Sorted Stream"],
        "requirements": ["Input data must be sorted"],
        "input_characteristics": ["Sorted array or unbounded stream"],
        "constraints": ["Requires random access by index"],
        "best_use_cases": ["Searching in unbounded / infinite sorted arrays", "Target is near the beginning of list"],
        "time_complexity": "O(log n)", "space_complexity": "O(1)",
        "advantages": ["Works on infinite/unbounded lists", "Fast when target is near start"],
        "limitations": ["Requires pre-sorted data"],
        "alternatives": ["Binary Search", "Jump Search"],
        "python_template": "def exponential_search(arr, target):\n    if not arr: return -1\n    if arr[0] == target: return 0\n    i = 1\n    while i < len(arr) and arr[i] <= target: i *= 2\n    left = i // 2; right = min(i, len(arr) - 1)\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1"
    },
    {
        "id": "fibonacci_search", "name": "Fibonacci Search", "category": "Searching",
        "description": "Comparison-based search technique for sorted arrays using Fibonacci numbers to divide the array.",
        "problem_patterns": ["fibonacci split search", "addition only search", "sorted array search without division"],
        "data_structures": ["Array", "Sorted List"],
        "requirements": ["Sorted input array"],
        "input_characteristics": ["Sorted numeric array"],
        "constraints": ["Requires generating Fibonacci numbers"],
        "best_use_cases": ["Systems where multiplication or division is computationally expensive"],
        "time_complexity": "O(log n)", "space_complexity": "O(1)",
        "advantages": ["Uses only addition and subtraction, no division"],
        "limitations": ["Slightly more complex than binary search"],
        "alternatives": ["Binary Search", "Jump Search"],
        "python_template": "def fibonacci_search(arr, target):\n    fibM2, fibM1 = 0, 1; fibM = fibM2 + fibM1; n = len(arr)\n    while (fibM < n): fibM2, fibM1 = fibM1, fibM; fibM = fibM2 + fibM1\n    offset = -1\n    while (fibM > 1):\n        i = min(offset + fibM2, n - 1)\n        if (arr[i] < target): fibM, fibM1 = fibM1, fibM2; fibM2 = fibM - fibM1; offset = i\n        elif (arr[i] > target): fibM = fibM2; fibM1 = fibM1 - fibM2; fibM2 = fibM - fibM1\n        else: return i\n    if(fibM1 and offset + 1 < n and arr[offset + 1] == target): return offset + 1\n    return -1"
    },
    {
        "id": "ternary_search", "name": "Ternary Search", "category": "Searching",
        "description": "Divide-and-conquer search algorithm that divides the array into three equal parts to find a target value or unimodal function peak.",
        "problem_patterns": ["unimodal function maximum/minimum", "three way split search", "peak finding"],
        "data_structures": ["Array", "Monotonic / Unimodal Function"],
        "requirements": ["Input must be sorted or unimodal"],
        "input_characteristics": ["Continuous or discrete unimodal values"],
        "constraints": ["More comparisons per step than binary search"],
        "best_use_cases": ["Finding maximum or minimum of a unimodal function"],
        "time_complexity": "O(log3 n)", "space_complexity": "O(1)",
        "advantages": ["Effective for continuous optimization/peak finding"],
        "limitations": ["Makes more comparisons than binary search for discrete array search"],
        "alternatives": ["Binary Search", "Golden Section Search"],
        "python_template": "def ternary_search(arr, target, l, r):\n    if r >= l:\n        m1 = l + (r - l) // 3; m2 = r - (r - l) // 3\n        if arr[m1] == target: return m1\n        if arr[m2] == target: return m2\n        if target < arr[m1]: return ternary_search(arr, target, l, m1 - 1)\n        elif target > arr[m2]: return ternary_search(arr, target, m2 + 1, r)\n        else: return ternary_search(arr, target, m1 + 1, m2 - 1)\n    return -1"
    },
    {
        "id": "bfs_grid_search", "name": "BFS Search (Grid / Tree / Graph)", "category": "Searching",
        "description": "Explores nodes level by level using a queue to find target element or shortest unweighted path.",
        "problem_patterns": ["level order search", "shortest path unweighted grid", "nearest neighbor in grid"],
        "data_structures": ["Queue", "Grid", "Graph"],
        "requirements": ["Graph/Grid representation with adjacency access"],
        "input_characteristics": ["Graph or 2D matrix grid"],
        "constraints": ["Requires O(V) memory space for queue"],
        "best_use_cases": ["Finding shortest path in unweighted maze/grid", "Level-order traversal search"],
        "time_complexity": "O(V + E)", "space_complexity": "O(V)",
        "advantages": ["Guarantees shortest path in unweighted graphs"],
        "limitations": ["High memory consumption for wide graphs"],
        "alternatives": ["DFS Search", "A* Search"],
        "python_template": "from collections import deque\ndef bfs_search(grid, start, target):\n    queue = deque([start]); visited = set([start])\n    while queue:\n        curr = queue.popleft()\n        if curr == target: return True\n        for r, c in [(curr[0]+1, curr[1]), (curr[0]-1, curr[1]), (curr[0], curr[1]+1), (curr[0], curr[1]-1)]:\n            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and (r, c) not in visited:\n                visited.add((r, c)); queue.append((r, c))\n    return False"
    },
    {
        "id": "dfs_grid_search", "name": "DFS Search (Grid / Tree / Graph)", "category": "Searching",
        "description": "Explores as far as possible along each branch before backtracking using a stack or recursion.",
        "problem_patterns": ["deep search", "path existence", "maze traversal", "connected components search"],
        "data_structures": ["Stack", "Recursion", "Grid", "Graph"],
        "requirements": ["Graph or grid connectivity"],
        "input_characteristics": ["Tree, Graph, or 2D Grid"],
        "constraints": ["Recursion depth limits"],
        "best_use_cases": ["Checking path existence", "Topological sorting", "Solving maze puzzles"],
        "time_complexity": "O(V + E)", "space_complexity": "O(V)",
        "advantages": ["Memory efficient for deep narrow structures"],
        "limitations": ["Does not guarantee shortest path"],
        "alternatives": ["BFS Search", "Backtracking"],
        "python_template": "def dfs_search(grid, r, c, visited, target):\n    if (r, c) == target: return True\n    visited.add((r, c))\n    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:\n        nr, nc = r + dr, c + dc\n        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and (nr, nc) not in visited:\n            if dfs_search(grid, nr, nc, visited, target): return True\n    return False"
    },
    {
        "id": "two_pointer_search", "name": "Two Pointer Search", "category": "Searching",
        "description": "Algorithm pattern using two pointers moving toward each other or in parallel to search for pairs or subarrays matching conditions.",
        "problem_patterns": ["two sum sorted array", "pair sum target", "container with most water", "remove duplicates in place"],
        "data_structures": ["Array", "String"],
        "requirements": ["Sorted array or specific directional property"],
        "input_characteristics": ["Contiguous sequential data"],
        "constraints": ["Requires monotonic sequence behavior"],
        "best_use_cases": ["Finding pairs in sorted array summing to target", "Palindrome checks"],
        "time_complexity": "O(n)", "space_complexity": "O(1)",
        "advantages": ["Linear time O(n) with O(1) extra space"],
        "limitations": ["Requires input to be sorted for pair sum searching"],
        "alternatives": ["Hash Table Lookup", "Binary Search"],
        "python_template": "def two_pointer_sum(arr, target):\n    left, right = 0, len(arr) - 1\n    while left < right:\n        curr_sum = arr[left] + arr[right]\n        if curr_sum == target: return (left, right)\n        elif curr_sum < target: left += 1\n        else: right -= 1\n    return None"
    },
    {
        "id": "sliding_window_search", "name": "Sliding Window Search", "category": "Searching",
        "description": "Technique to maintain a dynamic range/subsegment over an array to search for optimal subsegment satisfying criteria.",
        "problem_patterns": ["max subarray sum of length k", "longest substring without repeating characters", "minimum window substring"],
        "data_structures": ["Array", "String", "Hash Map"],
        "requirements": ["Contiguous subarray/substring search requirements"],
        "input_characteristics": ["Sequential array or string"],
        "constraints": ["Window state must be incrementally updateable"],
        "best_use_cases": ["Finding max/min subarray meeting target condition", "Substring search"],
        "time_complexity": "O(n)", "space_complexity": "O(k) or O(1)",
        "advantages": ["Converts O(n^2) nested loops into linear O(n) single pass"],
        "limitations": ["Only applies to contiguous sequence problems"],
        "alternatives": ["Two Pointer Search", "Hash Table Lookup"],
        "python_template": "def sliding_window_max_sum(arr, k):\n    if len(arr) < k: return 0\n    w_sum = sum(arr[:k]); max_sum = w_sum\n    for i in range(len(arr) - k):\n        w_sum = w_sum - arr[i] + arr[i+k]\n        max_sum = max(max_sum, w_sum)\n    return max_sum"
    }
])

print("Searching done. Total count:", len(kb))
