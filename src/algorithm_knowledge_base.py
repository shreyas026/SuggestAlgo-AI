"""
SuggestAlgo AI - Standardized Algorithm Knowledge Base Repository

Covers 10 primary categories with 10-15 structured algorithm entries per category.
"""

ALGORITHM_KNOWLEDGE_BASE = [
  {
    "id": "binary_search",
    "name": "Binary Search",
    "category": "Searching",
    "description": "Efficient search algorithm that finds the position of a target value within a sorted array by repeatedly dividing the search interval in half.",
    "problem_patterns": [
      "sorted array lookup",
      "find element in sorted list",
      "search in logarithmic time",
      "half search space",
      "find square root or threshold",
      "sorted data existence check"
    ],
    "data_structures": [
      "Array",
      "Sorted List"
    ],
    "requirements": [
      "Input array must be pre-sorted in ascending/descending order",
      "Random access by index O(1)"
    ],
    "input_characteristics": [
      "Sorted numerical or string sequence",
      "Known boundaries"
    ],
    "constraints": [
      "Data structure must support fast indexed access",
      "Requires prior sorting step"
    ],
    "best_use_cases": [
      "Searching in large sorted numerical arrays",
      "Finding boundaries/thresholds in monotonic functions",
      "Range lookup operations"
    ],
    "time_complexity": "O(log n)",
    "space_complexity": "O(1)",
    "advantages": [
      "Very fast for large sorted lists",
      "Minimal memory overhead"
    ],
    "limitations": [
      "Requires data to be pre-sorted",
      "Not suitable for linked lists"
    ],
    "alternatives": [
      "Interpolation Search",
      "Hash Table Lookup",
      "Exponential Search"
    ],
    "python_template": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1"
  },
  {
    "id": "linear_search",
    "name": "Linear Search",
    "category": "Searching",
    "description": "Sequential search algorithm that checks every element in a list one by one until a match is found or the end is reached.",
    "problem_patterns": [
      "unsorted list search",
      "find first occurrence",
      "small dataset lookup",
      "sequential search",
      "find match in unsorted array"
    ],
    "data_structures": [
      "Array",
      "Linked List",
      "Iterable"
    ],
    "requirements": [
      "None (works on unsorted or arbitrary data)"
    ],
    "input_characteristics": [
      "Unsorted or arbitrary order",
      "Small to medium size"
    ],
    "constraints": [
      "Inefficient for large arrays"
    ],
    "best_use_cases": [
      "Small collections",
      "Unsorted list single-pass lookup",
      "When sorting cost exceeds search cost"
    ],
    "time_complexity": "O(n)",
    "space_complexity": "O(1)",
    "advantages": [
      "Simple to implement",
      "No prerequisite sorting or index required"
    ],
    "limitations": [
      "Linear time lookup O(n) becomes slow for big datasets"
    ],
    "alternatives": [
      "Binary Search",
      "Hash Table Lookup"
    ],
    "python_template": "def linear_search(arr, target):\n    for idx, val in enumerate(arr):\n        if val == target: return idx\n    return -1"
  },
  {
    "id": "hash_table_lookup",
    "name": "Hash Table / Dictionary Lookup",
    "category": "Searching",
    "description": "Data structure technique that stores key-value pairs using a hash function for near-constant time lookups and insertions.",
    "problem_patterns": [
      "constant time lookup",
      "frequency count",
      "find duplicates",
      "two sum problem",
      "key value mapping",
      "fast index matching"
    ],
    "data_structures": [
      "Hash Map",
      "Hash Set",
      "Dictionary"
    ],
    "requirements": [
      "Keys must be hashable and immutable"
    ],
    "input_characteristics": [
      "Unique keys or items",
      "Unordered data"
    ],
    "constraints": [
      "Consumes extra memory for hash storage",
      "Potential hash collisions"
    ],
    "best_use_cases": [
      "Repeated fast lookups",
      "Element frequency counting",
      "Caching and memoization"
    ],
    "time_complexity": "O(1) average, O(n) worst case",
    "space_complexity": "O(n)",
    "advantages": [
      "Constant time average lookup O(1)",
      "Versatile key-value associations"
    ],
    "limitations": [
      "Unordered data storage",
      "Memory overhead"
    ],
    "alternatives": [
      "Binary Search Tree",
      "Trie"
    ],
    "python_template": "def hash_lookup(arr, target):\n    lookup_map = {val: idx for idx, val in enumerate(arr)}\n    return lookup_map.get(target, -1)"
  },
  {
    "id": "jump_search",
    "name": "Jump Search",
    "category": "Searching",
    "description": "Searching algorithm for sorted arrays that checks fewer elements than linear search by jumping ahead by fixed steps.",
    "problem_patterns": [
      "jump ahead search",
      "block search in sorted array",
      "sub-linear search without binary split"
    ],
    "data_structures": [
      "Array",
      "Sorted List"
    ],
    "requirements": [
      "Input array must be sorted"
    ],
    "input_characteristics": [
      "Sorted list",
      "Random access indexing"
    ],
    "constraints": [
      "Optimal step size is sqrt(n)"
    ],
    "best_use_cases": [
      "Searching sorted arrays when jumping backwards is expensive"
    ],
    "time_complexity": "O(sqrt(n))",
    "space_complexity": "O(1)",
    "advantages": [
      "Better than linear search",
      "Only jumps backward once"
    ],
    "limitations": [
      "Slower than binary search O(log n)"
    ],
    "alternatives": [
      "Binary Search",
      "Exponential Search"
    ],
    "python_template": "import math\ndef jump_search(arr, target):\n    n = len(arr); step = int(math.sqrt(n)); prev = 0\n    while arr[min(step, n)-1] < target:\n        prev = step; step += int(math.sqrt(n))\n        if prev >= n: return -1\n    while arr[prev] < target:\n        prev += 1\n        if prev == min(step, n): return -1\n    if arr[prev] == target: return prev\n    return -1"
  },
  {
    "id": "interpolation_search",
    "name": "Interpolation Search",
    "category": "Searching",
    "description": "Improved binary search variant for uniformly distributed sorted arrays that estimates position based on target value relative to low/high values.",
    "problem_patterns": [
      "uniformly distributed sorted search",
      "phonebook style search",
      "numeric range interpolation"
    ],
    "data_structures": [
      "Array",
      "Sorted List"
    ],
    "requirements": [
      "Data must be sorted and uniformly distributed"
    ],
    "input_characteristics": [
      "Sorted numbers with uniform intervals"
    ],
    "constraints": [
      "Performance degrades to O(n) if data is non-uniform"
    ],
    "best_use_cases": [
      "Searching in large uniformly distributed numerical sorted data"
    ],
    "time_complexity": "O(log log n) average, O(n) worst case",
    "space_complexity": "O(1)",
    "advantages": [
      "Sub-logarithmic search O(log log n) on uniform data"
    ],
    "limitations": [
      "Sensitive to non-uniform distribution"
    ],
    "alternatives": [
      "Binary Search",
      "Exponential Search"
    ],
    "python_template": "def interpolation_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high and target >= arr[low] and target <= arr[high]:\n        if low == high: return low if arr[low] == target else -1\n        pos = low + int(((high - low) / (arr[high] - arr[low])) * (target - arr[low]))\n        if arr[pos] == target: return pos\n        if arr[pos] < target: low = pos + 1\n        else: high = pos - 1\n    return -1"
  },
  {
    "id": "exponential_search",
    "name": "Exponential Search",
    "category": "Searching",
    "description": "Search algorithm for unbounded or infinite sorted lists that finds a range where target resides by doubling indices, then performs binary search.",
    "problem_patterns": [
      "unbounded array search",
      "infinite stream search",
      "unknown length sorted list"
    ],
    "data_structures": [
      "Array",
      "Sorted Stream"
    ],
    "requirements": [
      "Input data must be sorted"
    ],
    "input_characteristics": [
      "Sorted array or unbounded stream"
    ],
    "constraints": [
      "Requires random access by index"
    ],
    "best_use_cases": [
      "Searching in unbounded / infinite sorted arrays",
      "Target is near the beginning of list"
    ],
    "time_complexity": "O(log n)",
    "space_complexity": "O(1)",
    "advantages": [
      "Works on infinite/unbounded lists",
      "Fast when target is near start"
    ],
    "limitations": [
      "Requires pre-sorted data"
    ],
    "alternatives": [
      "Binary Search",
      "Jump Search"
    ],
    "python_template": "def exponential_search(arr, target):\n    if not arr: return -1\n    if arr[0] == target: return 0\n    i = 1\n    while i < len(arr) and arr[i] <= target: i *= 2\n    left = i // 2; right = min(i, len(arr) - 1)\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1"
  },
  {
    "id": "fibonacci_search",
    "name": "Fibonacci Search",
    "category": "Searching",
    "description": "Comparison-based search technique for sorted arrays using Fibonacci numbers to divide the array.",
    "problem_patterns": [
      "fibonacci split search",
      "addition only search",
      "sorted array search without division"
    ],
    "data_structures": [
      "Array",
      "Sorted List"
    ],
    "requirements": [
      "Sorted input array"
    ],
    "input_characteristics": [
      "Sorted numeric array"
    ],
    "constraints": [
      "Requires generating Fibonacci numbers"
    ],
    "best_use_cases": [
      "Systems where multiplication or division is computationally expensive"
    ],
    "time_complexity": "O(log n)",
    "space_complexity": "O(1)",
    "advantages": [
      "Uses only addition and subtraction, no division"
    ],
    "limitations": [
      "Slightly more complex than binary search"
    ],
    "alternatives": [
      "Binary Search",
      "Jump Search"
    ],
    "python_template": "def fibonacci_search(arr, target):\n    fibM2, fibM1 = 0, 1; fibM = fibM2 + fibM1; n = len(arr)\n    while (fibM < n): fibM2, fibM1 = fibM1, fibM; fibM = fibM2 + fibM1\n    offset = -1\n    while (fibM > 1):\n        i = min(offset + fibM2, n - 1)\n        if (arr[i] < target): fibM, fibM1 = fibM1, fibM2; fibM2 = fibM - fibM1; offset = i\n        elif (arr[i] > target): fibM = fibM2; fibM1 = fibM1 - fibM2; fibM2 = fibM - fibM1\n        else: return i\n    if(fibM1 and offset + 1 < n and arr[offset + 1] == target): return offset + 1\n    return -1"
  },
  {
    "id": "ternary_search",
    "name": "Ternary Search",
    "category": "Searching",
    "description": "Divide-and-conquer search algorithm that divides the array into three equal parts to find a target value or unimodal function peak.",
    "problem_patterns": [
      "unimodal function maximum/minimum",
      "three way split search",
      "peak finding"
    ],
    "data_structures": [
      "Array",
      "Monotonic / Unimodal Function"
    ],
    "requirements": [
      "Input must be sorted or unimodal"
    ],
    "input_characteristics": [
      "Continuous or discrete unimodal values"
    ],
    "constraints": [
      "More comparisons per step than binary search"
    ],
    "best_use_cases": [
      "Finding maximum or minimum of a unimodal function"
    ],
    "time_complexity": "O(log3 n)",
    "space_complexity": "O(1)",
    "advantages": [
      "Effective for continuous optimization/peak finding"
    ],
    "limitations": [
      "Makes more comparisons than binary search for discrete array search"
    ],
    "alternatives": [
      "Binary Search",
      "Golden Section Search"
    ],
    "python_template": "def ternary_search(arr, target, l, r):\n    if r >= l:\n        m1 = l + (r - l) // 3; m2 = r - (r - l) // 3\n        if arr[m1] == target: return m1\n        if arr[m2] == target: return m2\n        if target < arr[m1]: return ternary_search(arr, target, l, m1 - 1)\n        elif target > arr[m2]: return ternary_search(arr, target, m2 + 1, r)\n        else: return ternary_search(arr, target, m1 + 1, m2 - 1)\n    return -1"
  },
  {
    "id": "bfs_grid_search",
    "name": "BFS Search (Grid / Tree / Graph)",
    "category": "Searching",
    "description": "Explores nodes level by level using a queue to find target element or shortest unweighted path.",
    "problem_patterns": [
      "level order search",
      "shortest path unweighted grid",
      "nearest neighbor in grid"
    ],
    "data_structures": [
      "Queue",
      "Grid",
      "Graph"
    ],
    "requirements": [
      "Graph/Grid representation with adjacency access"
    ],
    "input_characteristics": [
      "Graph or 2D matrix grid"
    ],
    "constraints": [
      "Requires O(V) memory space for queue"
    ],
    "best_use_cases": [
      "Finding shortest path in unweighted maze/grid",
      "Level-order traversal search"
    ],
    "time_complexity": "O(V + E)",
    "space_complexity": "O(V)",
    "advantages": [
      "Guarantees shortest path in unweighted graphs"
    ],
    "limitations": [
      "High memory consumption for wide graphs"
    ],
    "alternatives": [
      "DFS Search",
      "A* Search"
    ],
    "python_template": "from collections import deque\ndef bfs_search(grid, start, target):\n    queue = deque([start]); visited = set([start])\n    while queue:\n        curr = queue.popleft()\n        if curr == target: return True\n        for r, c in [(curr[0]+1, curr[1]), (curr[0]-1, curr[1]), (curr[0], curr[1]+1), (curr[0], curr[1]-1)]:\n            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and (r, c) not in visited:\n                visited.add((r, c)); queue.append((r, c))\n    return False"
  },
  {
    "id": "dfs_grid_search",
    "name": "DFS Search (Grid / Tree / Graph)",
    "category": "Searching",
    "description": "Explores as far as possible along each branch before backtracking using a stack or recursion.",
    "problem_patterns": [
      "deep search",
      "path existence",
      "maze traversal",
      "connected components search"
    ],
    "data_structures": [
      "Stack",
      "Recursion",
      "Grid",
      "Graph"
    ],
    "requirements": [
      "Graph or grid connectivity"
    ],
    "input_characteristics": [
      "Tree, Graph, or 2D Grid"
    ],
    "constraints": [
      "Recursion depth limits"
    ],
    "best_use_cases": [
      "Checking path existence",
      "Topological sorting",
      "Solving maze puzzles"
    ],
    "time_complexity": "O(V + E)",
    "space_complexity": "O(V)",
    "advantages": [
      "Memory efficient for deep narrow structures"
    ],
    "limitations": [
      "Does not guarantee shortest path"
    ],
    "alternatives": [
      "BFS Search",
      "Backtracking"
    ],
    "python_template": "def dfs_search(grid, r, c, visited, target):\n    if (r, c) == target: return True\n    visited.add((r, c))\n    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:\n        nr, nc = r + dr, c + dc\n        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and (nr, nc) not in visited:\n            if dfs_search(grid, nr, nc, visited, target): return True\n    return False"
  },
  {
    "id": "two_pointer_search",
    "name": "Two Pointer Search",
    "category": "Searching",
    "description": "Algorithm pattern using two pointers moving toward each other or in parallel to search for pairs or subarrays matching conditions.",
    "problem_patterns": [
      "two sum sorted array",
      "pair sum target",
      "container with most water",
      "remove duplicates in place"
    ],
    "data_structures": [
      "Array",
      "String"
    ],
    "requirements": [
      "Sorted array or specific directional property"
    ],
    "input_characteristics": [
      "Contiguous sequential data"
    ],
    "constraints": [
      "Requires monotonic sequence behavior"
    ],
    "best_use_cases": [
      "Finding pairs in sorted array summing to target",
      "Palindrome checks"
    ],
    "time_complexity": "O(n)",
    "space_complexity": "O(1)",
    "advantages": [
      "Linear time O(n) with O(1) extra space"
    ],
    "limitations": [
      "Requires input to be sorted for pair sum searching"
    ],
    "alternatives": [
      "Hash Table Lookup",
      "Binary Search"
    ],
    "python_template": "def two_pointer_sum(arr, target):\n    left, right = 0, len(arr) - 1\n    while left < right:\n        curr_sum = arr[left] + arr[right]\n        if curr_sum == target: return (left, right)\n        elif curr_sum < target: left += 1\n        else: right -= 1\n    return None"
  },
  {
    "id": "sliding_window_search",
    "name": "Sliding Window Search",
    "category": "Searching",
    "description": "Technique to maintain a dynamic range/subsegment over an array to search for optimal subsegment satisfying criteria.",
    "problem_patterns": [
      "max subarray sum of length k",
      "longest substring without repeating characters",
      "minimum window substring"
    ],
    "data_structures": [
      "Array",
      "String",
      "Hash Map"
    ],
    "requirements": [
      "Contiguous subarray/substring search requirements"
    ],
    "input_characteristics": [
      "Sequential array or string"
    ],
    "constraints": [
      "Window state must be incrementally updateable"
    ],
    "best_use_cases": [
      "Finding max/min subarray meeting target condition",
      "Substring search"
    ],
    "time_complexity": "O(n)",
    "space_complexity": "O(k) or O(1)",
    "advantages": [
      "Converts O(n^2) nested loops into linear O(n) single pass"
    ],
    "limitations": [
      "Only applies to contiguous sequence problems"
    ],
    "alternatives": [
      "Two Pointer Search",
      "Hash Table Lookup"
    ],
    "python_template": "def sliding_window_max_sum(arr, k):\n    if len(arr) < k: return 0\n    w_sum = sum(arr[:k]); max_sum = w_sum\n    for i in range(len(arr) - k):\n        w_sum = w_sum - arr[i] + arr[i+k]\n        max_sum = max(max_sum, w_sum)\n    return max_sum"
  },
  {
    "id": "bubble_sort",
    "name": "Bubble Sort",
    "category": "Sorting",
    "description": "Simple comparison sort that repeatedly steps through the list, compares adjacent elements and swaps them if in wrong order.",
    "problem_patterns": [
      "simple sorting",
      "adjacent element swaps",
      "educational sort"
    ],
    "data_structures": [
      "Array"
    ],
    "requirements": [
      "Comparable elements"
    ],
    "input_characteristics": [
      "Small list or nearly sorted list"
    ],
    "constraints": [
      "Quadratic time complexity O(n^2)"
    ],
    "best_use_cases": [
      "Educational purposes",
      "Tiny arrays",
      "Detecting if array is already sorted"
    ],
    "time_complexity": "O(n^2) worst/avg, O(n) best",
    "space_complexity": "O(1)",
    "advantages": [
      "In-place",
      "Stable",
      "Simple"
    ],
    "limitations": [
      "Very slow for large arrays"
    ],
    "alternatives": [
      "Insertion Sort",
      "Quick Sort"
    ],
    "python_template": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        swapped = False\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]; swapped = True\n        if not swapped: break\n    return arr"
  },
  {
    "id": "selection_sort",
    "name": "Selection Sort",
    "category": "Sorting",
    "description": "In-place comparison sort that divides array into sorted and unsorted parts, repeatedly selecting the minimum from unsorted part.",
    "problem_patterns": [
      "minimize memory writes",
      "select minimum element repeatedly"
    ],
    "data_structures": [
      "Array"
    ],
    "requirements": [
      "Comparable items"
    ],
    "input_characteristics": [
      "Small array size"
    ],
    "constraints": [
      "Always O(n^2) comparisons regardless of input"
    ],
    "best_use_cases": [
      "Small datasets where memory writing is expensive"
    ],
    "time_complexity": "O(n^2)",
    "space_complexity": "O(1)",
    "advantages": [
      "Makes minimum number of memory writes O(n)",
      "In-place"
    ],
    "limitations": [
      "Unstable sort",
      "Slow O(n^2)"
    ],
    "alternatives": [
      "Insertion Sort",
      "Heap Sort"
    ],
    "python_template": "def selection_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        min_idx = i\n        for j in range(i+1, n):\n            if arr[j] < arr[min_idx]: min_idx = j\n        arr[i], arr[min_idx] = arr[min_idx], arr[i]\n    return arr"
  },
  {
    "id": "insertion_sort",
    "name": "Insertion Sort",
    "category": "Sorting",
    "description": "Simple sorting algorithm that builds the final sorted array one item at a time, highly efficient for small or nearly sorted data.",
    "problem_patterns": [
      "nearly sorted array sort",
      "online stream sorting",
      "small subarray sort"
    ],
    "data_structures": [
      "Array",
      "Linked List"
    ],
    "requirements": [
      "Comparable items"
    ],
    "input_characteristics": [
      "Small dataset or almost sorted data"
    ],
    "constraints": [
      "O(n^2) for reverse ordered inputs"
    ],
    "best_use_cases": [
      "Sorting small lists (n < 25)",
      "Online sorting of incoming items"
    ],
    "time_complexity": "O(n^2) worst, O(n) best",
    "space_complexity": "O(1)",
    "advantages": [
      "Adaptive (O(n) on nearly sorted data)",
      "Stable",
      "In-place"
    ],
    "limitations": [
      "High comparison cost on large unsorted arrays"
    ],
    "alternatives": [
      "TimSort",
      "Merge Sort"
    ],
    "python_template": "def insertion_sort(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]; j = i - 1\n        while j >= 0 and arr[j] > key:\n            arr[j + 1] = arr[j]; j -= 1\n        arr[j + 1] = key\n    return arr"
  },
  {
    "id": "merge_sort",
    "name": "Merge Sort",
    "category": "Sorting",
    "description": "Divide-and-conquer algorithm that recursively splits array into halves, sorts them, and merges sorted halves.",
    "problem_patterns": [
      "guaranteed O(n log n) sort",
      "external sorting",
      "linked list sorting",
      "stable sorting"
    ],
    "data_structures": [
      "Array",
      "Linked List"
    ],
    "requirements": [
      "Requires O(n) auxiliary space for arrays"
    ],
    "input_characteristics": [
      "General large datasets"
    ],
    "constraints": [
      "Auxiliary space O(n)"
    ],
    "best_use_cases": [
      "Guaranteed O(n log n) sorting",
      "Sorting linked lists",
      "External sorting of big files"
    ],
    "time_complexity": "O(n log n)",
    "space_complexity": "O(n)",
    "advantages": [
      "Stable sort",
      "Guaranteed O(n log n) performance",
      "Parallelizable"
    ],
    "limitations": [
      "Requires extra O(n) space"
    ],
    "alternatives": [
      "Quick Sort",
      "TimSort"
    ],
    "python_template": "def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid]); right = merge_sort(arr[mid:])\n    res = []; i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]: res.append(left[i]); i += 1\n        else: res.append(right[j]); j += 1\n    res.extend(left[i:]); res.extend(right[j:])\n    return res"
  },
  {
    "id": "quick_sort",
    "name": "Quick Sort",
    "category": "Sorting",
    "description": "In-place divide-and-conquer sort selecting a pivot element and partitioning array into smaller and larger sub-arrays.",
    "problem_patterns": [
      "fast general sort",
      "in place partitioning",
      "cache friendly sorting"
    ],
    "data_structures": [
      "Array"
    ],
    "requirements": [
      "Random access memory array"
    ],
    "input_characteristics": [
      "General unsorted array"
    ],
    "constraints": [
      "Worst-case O(n^2) if pivot selection is poor"
    ],
    "best_use_cases": [
      "Fast in-place general purpose sorting"
    ],
    "time_complexity": "O(n log n) average, O(n^2) worst case",
    "space_complexity": "O(log n) stack space",
    "advantages": [
      "Fast in practice",
      "In-place",
      "Excellent cache locality"
    ],
    "limitations": [
      "Unstable sort",
      "Worst-case quadratic time"
    ],
    "alternatives": [
      "Merge Sort",
      "Heap Sort",
      "TimSort"
    ],
    "python_template": "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)"
  },
  {
    "id": "heap_sort",
    "name": "Heap Sort",
    "category": "Sorting",
    "description": "Comparison-based sorting technique based on Binary Heap data structure, finding maximum and placing it at the end.",
    "problem_patterns": [
      "in place O(n log n) sort",
      "priority queue sort",
      "top k elements sort"
    ],
    "data_structures": [
      "Binary Heap",
      "Array"
    ],
    "requirements": [
      "Comparable array elements"
    ],
    "input_characteristics": [
      "General numerical or comparable arrays"
    ],
    "constraints": [
      "Poor cache locality due to heap jumps"
    ],
    "best_use_cases": [
      "Systems requiring guaranteed O(n log n) with O(1) auxiliary space"
    ],
    "time_complexity": "O(n log n)",
    "space_complexity": "O(1)",
    "advantages": [
      "Guaranteed O(n log n)",
      "In-place O(1) extra space"
    ],
    "limitations": [
      "Unstable sort",
      "Slower than QuickSort in practice"
    ],
    "alternatives": [
      "Quick Sort",
      "Merge Sort"
    ],
    "python_template": "import heapq\ndef heap_sort(arr):\n    heapq.heapify(arr)\n    return [heapq.heappop(arr) for _ in range(len(arr))]"
  },
  {
    "id": "counting_sort",
    "name": "Counting Sort",
    "category": "Sorting",
    "description": "Non-comparison integer sorting algorithm operating by counting number of objects having distinct key values.",
    "problem_patterns": [
      "integer array sort",
      "small key range sorting",
      "linear time non comparison sort"
    ],
    "data_structures": [
      "Array",
      "Frequency Counter"
    ],
    "requirements": [
      "Keys must be non-negative integers in small range K"
    ],
    "input_characteristics": [
      "Integer values with known max range"
    ],
    "constraints": [
      "Space consuming if range K is much larger than N"
    ],
    "best_use_cases": [
      "Sorting discrete integers when range K is O(N)"
    ],
    "time_complexity": "O(n + k)",
    "space_complexity": "O(n + k)",
    "advantages": [
      "Linear time O(n+k)",
      "Stable sort"
    ],
    "limitations": [
      "Only works on integers or discrete keys",
      "High space if K is large"
    ],
    "alternatives": [
      "Radix Sort",
      "Bucket Sort"
    ],
    "python_template": "def counting_sort(arr):\n    if not arr: return arr\n    max_val = max(arr); count = [0] * (max_val + 1)\n    for num in arr: count[num] += 1\n    sorted_arr = []\n    for val, freq in enumerate(count): sorted_arr.extend([val] * freq)\n    return sorted_arr"
  },
  {
    "id": "radix_sort",
    "name": "Radix Sort",
    "category": "Sorting",
    "description": "Non-comparative integer sorting algorithm that sorts data with integer keys by grouping keys by individual digits.",
    "problem_patterns": [
      "digit by digit sorting",
      "large integer list sort",
      "fixed length string sort"
    ],
    "data_structures": [
      "Array",
      "Buckets"
    ],
    "requirements": [
      "Keys must be expressible in digit/character format"
    ],
    "input_characteristics": [
      "Integers or fixed-length strings"
    ],
    "constraints": [
      "Requires stable underlying digit sort"
    ],
    "best_use_cases": [
      "Sorting large lists of integers or fixed-length strings in O(nk)"
    ],
    "time_complexity": "O(d * (n + k))",
    "space_complexity": "O(n + k)",
    "advantages": [
      "Faster than O(n log n) comparison sorts for fixed digit length"
    ],
    "limitations": [
      "Not general purpose, tied to key representations"
    ],
    "alternatives": [
      "Counting Sort",
      "Bucket Sort"
    ],
    "python_template": "def radix_sort(arr):\n    if not arr: return arr\n    max1 = max(arr); exp = 1; output = [0] * len(arr)\n    while max1 // exp > 0:\n        count = [0] * 10\n        for i in range(len(arr)): count[(arr[i] // exp) % 10] += 1\n        for i in range(1, 10): count[i] += count[i - 1]\n        i = len(arr) - 1\n        while i >= 0:\n            idx = (arr[i] // exp) % 10\n            output[count[idx] - 1] = arr[i]; count[idx] -= 1; i -= 1\n        for i in range(len(arr)): arr[i] = output[i]\n        exp *= 10\n    return arr"
  },
  {
    "id": "bucket_sort",
    "name": "Bucket Sort",
    "category": "Sorting",
    "description": "Distribution sort that divides array into multiple buckets, then sorts individual buckets independently.",
    "problem_patterns": [
      "floating point uniform distribution sort",
      "bucket partition sort"
    ],
    "data_structures": [
      "Array of Lists"
    ],
    "requirements": [
      "Input should be uniformly distributed over an interval"
    ],
    "input_characteristics": [
      "Floating point numbers in range [0, 1)"
    ],
    "constraints": [
      "Worst-case O(n^2) if items cluster into one bucket"
    ],
    "best_use_cases": [
      "Sorting floating point numbers uniformly distributed across a range"
    ],
    "time_complexity": "O(n + k) average, O(n^2) worst case",
    "space_complexity": "O(n)",
    "advantages": [
      "Linear time O(n) average on uniform input"
    ],
    "limitations": [
      "Performance degrades if data is heavily clustered"
    ],
    "alternatives": [
      "Radix Sort",
      "Quick Sort"
    ],
    "python_template": "def bucket_sort(arr):\n    if not arr: return arr\n    n = len(arr); buckets = [[] for _ in range(n)]\n    for num in arr:\n        idx = int(num * n)\n        buckets[min(idx, n - 1)].append(num)\n    for i in range(n): buckets[i].sort()\n    res = []\n    for b in buckets: res.extend(b)\n    return res"
  },
  {
    "id": "shell_sort",
    "name": "Shell Sort",
    "category": "Sorting",
    "description": "In-place comparison sort generalization of insertion sort that allows exchanges of items that are far apart.",
    "problem_patterns": [
      "gap insertion sort",
      "diminishing increment sort"
    ],
    "data_structures": [
      "Array"
    ],
    "requirements": [
      "Comparable items"
    ],
    "input_characteristics": [
      "Medium size arrays"
    ],
    "constraints": [
      "Complexity depends heavily on gap sequence selection"
    ],
    "best_use_cases": [
      "Embedded systems where code size must be minimal and memory limited"
    ],
    "time_complexity": "O(n^(4/3)) to O(n^2)",
    "space_complexity": "O(1)",
    "advantages": [
      "In-place O(1) space",
      "Faster than insertion sort for medium arrays"
    ],
    "limitations": [
      "Unstable sort",
      "Sub-optimal for huge arrays"
    ],
    "alternatives": [
      "Quick Sort",
      "Insertion Sort"
    ],
    "python_template": "def shell_sort(arr):\n    n = len(arr); gap = n // 2\n    while gap > 0:\n        for i in range(gap, n):\n            temp = arr[i]; j = i\n            while j >= gap and arr[j - gap] > temp:\n                arr[j] = arr[j - gap]; j -= gap\n            arr[j] = temp\n        gap //= 2\n    return arr"
  },
  {
    "id": "timsort",
    "name": "TimSort",
    "category": "Sorting",
    "description": "Hybrid stable sorting algorithm derived from merge sort and insertion sort, standard default sort in Python and Java.",
    "problem_patterns": [
      "default language sort",
      "real world mixed data sort",
      "stable hybrid sort"
    ],
    "data_structures": [
      "Array"
    ],
    "requirements": [
      "Comparable items"
    ],
    "input_characteristics": [
      "Real world data with existing run segments"
    ],
    "constraints": [
      "Auxiliary space O(n)"
    ],
    "best_use_cases": [
      "General purpose default sorting for real-world heterogeneous datasets"
    ],
    "time_complexity": "O(n log n) worst, O(n) best",
    "space_complexity": "O(n)",
    "advantages": [
      "Extremely fast on real-world data with natural runs",
      "Stable",
      "Adaptive"
    ],
    "limitations": [
      "Requires auxiliary memory O(n)"
    ],
    "alternatives": [
      "Merge Sort",
      "Quick Sort"
    ],
    "python_template": "def timsort(arr):\n    return sorted(arr)"
  },
  {
    "id": "logistic_regression",
    "name": "Logistic Regression",
    "category": "Classification",
    "description": "Linear classification model that predicts class probability using sigmoid function.",
    "problem_patterns": [
      "binary classification baseline",
      "linear decision boundary",
      "predict outcome probability",
      "churn prediction baseline",
      "spam classification baseline"
    ],
    "data_structures": [
      "Feature Matrix",
      "Target Vector"
    ],
    "requirements": [
      "Numerical or one-hot encoded features",
      "No missing values"
    ],
    "input_characteristics": [
      "Linearly separable tabular data"
    ],
    "constraints": [
      "Assumes linear independence of log-odds"
    ],
    "best_use_cases": [
      "Binary classification baseline",
      "Credit scoring",
      "Medical diagnosis probability"
    ],
    "time_complexity": "O(n * d)",
    "space_complexity": "O(d)",
    "advantages": [
      "Highly interpretable coefficients",
      "Outputs calibrated probabilities"
    ],
    "limitations": [
      "Cannot model complex non-linear relationships without feature engineering"
    ],
    "alternatives": [
      "Decision Tree Classifier",
      "Support Vector Machine"
    ],
    "python_template": "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression()\nmodel.fit(X_train, y_train)\npreds = model.predict(X_test)"
  },
  {
    "id": "decision_tree_classifier",
    "name": "Decision Tree Classifier",
    "category": "Classification",
    "description": "Non-parametric supervised learning algorithm that creates tree of decision rules for classification.",
    "problem_patterns": [
      "interpretable rule tree",
      "non linear decision boundary",
      "categorical feature classification"
    ],
    "data_structures": [
      "Binary / Multi-way Tree"
    ],
    "requirements": [
      "Preprocessed feature matrix"
    ],
    "input_characteristics": [
      "Tabular data with mixed categorical/numerical features"
    ],
    "constraints": [
      "Prone to overfitting if unconstrained"
    ],
    "best_use_cases": [
      "Rule-based business decisions",
      "Customer segmentation",
      "Medical diagnostic trees"
    ],
    "time_complexity": "O(d * n log n) train",
    "space_complexity": "O(nodes)",
    "advantages": [
      "Human readable decision path",
      "Handles non-linear relationships natively"
    ],
    "limitations": [
      "High variance, prone to overfitting"
    ],
    "alternatives": [
      "Random Forest Classifier",
      "Extra Trees Classifier"
    ],
    "python_template": "from sklearn.tree import DecisionTreeClassifier\nmodel = DecisionTreeClassifier(max_depth=5)\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "random_forest_classifier",
    "name": "Random Forest Classifier",
    "category": "Classification",
    "description": "Ensemble learning method operating by constructing multiple decision trees and bagging outputs.",
    "problem_patterns": [
      "robust tabular classification",
      "feature importance ranking",
      "high accuracy ensemble"
    ],
    "data_structures": [
      "Forest of Decision Trees"
    ],
    "requirements": [
      "Numeric feature matrix"
    ],
    "input_characteristics": [
      "Tabular datasets with complex interactions"
    ],
    "constraints": [
      "Slower inference than single tree"
    ],
    "best_use_cases": [
      "General tabular classification benchmark",
      "Customer churn prediction",
      "Fraud detection"
    ],
    "time_complexity": "O(n_trees * d * n log n)",
    "space_complexity": "O(n_trees * nodes)",
    "advantages": [
      "High predictive accuracy",
      "Resistant to overfitting",
      "Provides feature importances"
    ],
    "limitations": [
      "Black-box model compared to single tree"
    ],
    "alternatives": [
      "Gradient Boosting Classifier",
      "XGBoost Classifier"
    ],
    "python_template": "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100)\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "extra_trees_classifier",
    "name": "Extra Trees Classifier",
    "category": "Classification",
    "description": "Extremely Randomized Trees classifier that randomizes tree splits further to reduce variance.",
    "problem_patterns": [
      "extremely randomized trees classification",
      "fast ensemble classification"
    ],
    "data_structures": [
      "Forest of Decision Trees"
    ],
    "requirements": [
      "Numeric features"
    ],
    "input_characteristics": [
      "Tabular features with noise"
    ],
    "constraints": [
      "Can create larger trees than Random Forest"
    ],
    "best_use_cases": [
      "Fast ensemble training on noisy tabular data"
    ],
    "time_complexity": "O(n_trees * d * n)",
    "space_complexity": "O(n_trees * nodes)",
    "advantages": [
      "Faster training than Random Forest",
      "Reduces variance"
    ],
    "limitations": [
      "Slightly higher bias"
    ],
    "alternatives": [
      "Random Forest Classifier",
      "Gradient Boosting Classifier"
    ],
    "python_template": "from sklearn.ensemble import ExtraTreesClassifier\nmodel = ExtraTreesClassifier(n_estimators=100)\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "gradient_boosting_classifier",
    "name": "Gradient Boosting Classifier",
    "category": "Classification",
    "description": "Ensemble technique that builds decision trees sequentially, each correcting errors of previous trees.",
    "problem_patterns": [
      "boosting classifier",
      "sequential error correction",
      "top competition classification model"
    ],
    "data_structures": [
      "Sequence of Trees"
    ],
    "requirements": [
      "Numerical features"
    ],
    "input_characteristics": [
      "Tabular data requiring high predictive accuracy"
    ],
    "constraints": [
      "Requires tuning hyper-parameters (learning rate, depth)"
    ],
    "best_use_cases": [
      "Competitive machine learning benchmarks",
      "High precision classification tasks"
    ],
    "time_complexity": "O(n_trees * d * n log n)",
    "space_complexity": "O(n_trees * nodes)",
    "advantages": [
      "Often achieves best predictive accuracy on tabular data"
    ],
    "limitations": [
      "Prone to overfitting if learning rate is too high"
    ],
    "alternatives": [
      "XGBoost Classifier",
      "AdaBoost Classifier"
    ],
    "python_template": "from sklearn.ensemble import GradientBoostingClassifier\nmodel = GradientBoostingClassifier()\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "adaboost_classifier",
    "name": "AdaBoost Classifier",
    "category": "Classification",
    "description": "Adaptive Boosting ensemble algorithm that adjusts weights of misclassified instances iteratively.",
    "problem_patterns": [
      "adaptive boosting",
      "weighted sample classification"
    ],
    "data_structures": [
      "Sequence of Decision Stumps"
    ],
    "requirements": [
      "Numeric features"
    ],
    "input_characteristics": [
      "Tabular datasets"
    ],
    "constraints": [
      "Sensitive to noisy data and outliers"
    ],
    "best_use_cases": [
      "Boosting weak learners for binary classification"
    ],
    "time_complexity": "O(n_estimators * n * d)",
    "space_complexity": "O(n_estimators)",
    "advantages": [
      "Simple boosting concept",
      "Resistant to overfitting on clean data"
    ],
    "limitations": [
      "Sensitive to noisy data and outliers"
    ],
    "alternatives": [
      "Gradient Boosting Classifier",
      "Random Forest Classifier"
    ],
    "python_template": "from sklearn.ensemble import AdaBoostClassifier\nmodel = AdaBoostClassifier()\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "support_vector_classifier",
    "name": "Support Vector Classifier (SVC)",
    "category": "Classification",
    "description": "Classifier that finds hyperplanes maximizing margin between classes, using kernel tricks for non-linear boundaries.",
    "problem_patterns": [
      "maximum margin classifier",
      "kernel trick classification",
      "high dimensional classification"
    ],
    "data_structures": [
      "Support Vectors"
    ],
    "requirements": [
      "Scaled numerical features"
    ],
    "input_characteristics": [
      "High dimensional data, small to medium sample size"
    ],
    "constraints": [
      "Scales quadratically O(n^2) with sample size"
    ],
    "best_use_cases": [
      "Text classification",
      "Bioinformatics",
      "Medium dataset high-dimensional boundary matching"
    ],
    "time_complexity": "O(n^2 * d) to O(n^3)",
    "space_complexity": "O(n_support_vectors * d)",
    "advantages": [
      "Effective in high dimensional spaces",
      "Memory efficient using support vectors"
    ],
    "limitations": [
      "Slow training on large datasets (n > 50,000)"
    ],
    "alternatives": [
      "Logistic Regression",
      "Random Forest Classifier"
    ],
    "python_template": "from sklearn.svm import SVC\nmodel = SVC(kernel='rbf')\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "knn_classifier",
    "name": "K-Nearest Neighbors Classifier (KNN)",
    "category": "Classification",
    "description": "Non-parametric lazy learning classifier that assigns class based on majority vote of K nearest samples.",
    "problem_patterns": [
      "lazy learning classification",
      "distance based neighbor vote",
      "similarity search classification"
    ],
    "data_structures": [
      "KD-Tree",
      "Ball-Tree",
      "Feature Matrix"
    ],
    "requirements": [
      "Feature scaling (StandardScaler/MinMaxScaler)"
    ],
    "input_characteristics": [
      "Small to medium tabular data with low dimensions"
    ],
    "constraints": [
      "High prediction memory/time O(n * d)"
    ],
    "best_use_cases": [
      "Instance-based learning",
      "Recommendation systems",
      "Simple baseline"
    ],
    "time_complexity": "O(n * d) query",
    "space_complexity": "O(n * d)",
    "advantages": [
      "No explicit training step",
      "Naturally handles multi-class classification"
    ],
    "limitations": [
      "Slow prediction phase",
      "Sensitive to scale and irrelevant features"
    ],
    "alternatives": [
      "Support Vector Classifier",
      "Random Forest Classifier"
    ],
    "python_template": "from sklearn.neighbors import KNeighborsClassifier\nmodel = KNeighborsClassifier(n_neighbors=5)\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "naive_bayes_classifier",
    "name": "Naive Bayes Classifier",
    "category": "Classification",
    "description": "Probabilistic classifier based on Bayes Theorem assuming strong (naive) feature independence.",
    "problem_patterns": [
      "fast probabilistic text classification",
      "spam filtering naive bayes",
      "document classification"
    ],
    "data_structures": [
      "Probability Tables"
    ],
    "requirements": [
      "Feature independence assumption"
    ],
    "input_characteristics": [
      "Text word counts, categorical or continuous sparse features"
    ],
    "constraints": [
      "Zero frequency problem without smoothing"
    ],
    "best_use_cases": [
      "Spam detection",
      "Text document categorization",
      "Real-time probabilistic predictions"
    ],
    "time_complexity": "O(n * d)",
    "space_complexity": "O(classes * d)",
    "advantages": [
      "Extremely fast training and inference",
      "Works well with high dimensional sparse text"
    ],
    "limitations": [
      "Feature independence assumption rarely holds strictly"
    ],
    "alternatives": [
      "Logistic Regression",
      "Support Vector Classifier"
    ],
    "python_template": "from sklearn.naive_bayes import GaussianNB\nmodel = GaussianNB()\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "sgd_classifier",
    "name": "SGD Classifier",
    "category": "Classification",
    "description": "Linear classifiers (SVM, Logistic Regression) optimized using Stochastic Gradient Descent.",
    "problem_patterns": [
      "large scale online classification",
      "streaming data classifier",
      "out of core learning"
    ],
    "data_structures": [
      "Weight Vector"
    ],
    "requirements": [
      "Scaled features"
    ],
    "input_characteristics": [
      "Very large tabular datasets (n > 100,000) or streaming batches"
    ],
    "constraints": [
      "Sensitive to feature scaling and learning rate"
    ],
    "best_use_cases": [
      "Large-scale linear classification",
      "Online learning streams"
    ],
    "time_complexity": "O(n * d)",
    "space_complexity": "O(d)",
    "advantages": [
      "Efficient for huge datasets",
      "Supports online learning"
    ],
    "limitations": [
      "Requires hyper-parameter tuning (learning rate schedule)"
    ],
    "alternatives": [
      "Logistic Regression",
      "Linear SVC"
    ],
    "python_template": "from sklearn.linear_model import SGDClassifier\nmodel = SGDClassifier(loss='log_loss')\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "xgboost_classifier",
    "name": "XGBoost Classifier",
    "category": "Classification",
    "description": "Optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.",
    "problem_patterns": [
      "extreme gradient boosting",
      "kaggle winning classification",
      "regularized tree boosting"
    ],
    "data_structures": [
      "DMatrix",
      "Gradient Trees"
    ],
    "requirements": [
      "Numeric feature matrix"
    ],
    "input_characteristics": [
      "Structured tabular datasets"
    ],
    "constraints": [
      "High hyper-parameter tuning requirements"
    ],
    "best_use_cases": [
      "Tabular classification competitions",
      "Production high-performance classification"
    ],
    "time_complexity": "O(n_trees * d * n log n)",
    "space_complexity": "O(n_trees * nodes)",
    "advantages": [
      "Built-in regularization",
      "Fast multi-threading",
      "Handles missing values automatically"
    ],
    "limitations": [
      "Complex hyper-parameter tuning space"
    ],
    "alternatives": [
      "Gradient Boosting Classifier",
      "LightGBM Classifier"
    ],
    "python_template": "import xgboost as xgb\nmodel = xgb.XGBClassifier()\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "linear_regression",
    "name": "Linear Regression",
    "category": "Regression",
    "description": "Fundamental regression model fitting a linear relationship between features and continuous target variable.",
    "problem_patterns": [
      "linear continuous target prediction",
      "house price baseline",
      "sales forecasting linear"
    ],
    "data_structures": [
      "Feature Matrix",
      "Coefficient Vector"
    ],
    "requirements": [
      "Continuous numeric target",
      "Scaled features"
    ],
    "input_characteristics": [
      "Linear tabular features"
    ],
    "constraints": [
      "Sensitive to outliers and multicollinearity"
    ],
    "best_use_cases": [
      "Simple continuous target forecasting",
      "Trend estimation",
      "Baseline regression"
    ],
    "time_complexity": "O(n * d^2)",
    "space_complexity": "O(d)",
    "advantages": [
      "Extremely simple and interpretable",
      "Fast analytical solution"
    ],
    "limitations": [
      "Underfits non-linear data"
    ],
    "alternatives": [
      "Ridge Regression",
      "Decision Tree Regressor"
    ],
    "python_template": "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "ridge_regression",
    "name": "Ridge Regression",
    "category": "Regression",
    "description": "L2 regularized linear regression that prevents overfitting by shrinking feature coefficients.",
    "problem_patterns": [
      "l2 regularization regression",
      "multicollinearity prevention",
      "regularized linear model"
    ],
    "data_structures": [
      "Coefficient Vector"
    ],
    "requirements": [
      "Scaled numeric features"
    ],
    "input_characteristics": [
      "Tabular regression with correlated features"
    ],
    "constraints": [
      "Does not perform feature selection (coefficients shrink near zero, not zero)"
    ],
    "best_use_cases": [
      "Continuous regression with correlated predictors"
    ],
    "time_complexity": "O(n * d^2)",
    "space_complexity": "O(d)",
    "advantages": [
      "Solves multicollinearity issues",
      "Reduces model variance"
    ],
    "limitations": [
      "Keeps all features"
    ],
    "alternatives": [
      "Lasso Regression",
      "Elastic Net Regression"
    ],
    "python_template": "from sklearn.linear_model import Ridge\nmodel = Ridge(alpha=1.0)\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "lasso_regression",
    "name": "Lasso Regression",
    "category": "Regression",
    "description": "L1 regularized linear regression performing feature selection by driving irrelevant feature coefficients strictly to zero.",
    "problem_patterns": [
      "l1 regularization regression",
      "automatic feature selection",
      "sparse linear regression"
    ],
    "data_structures": [
      "Coefficient Vector"
    ],
    "requirements": [
      "Scaled features"
    ],
    "input_characteristics": [
      "High dimensional regression data with sparse relevant features"
    ],
    "constraints": [
      "Selects at most n features when d > n"
    ],
    "best_use_cases": [
      "Feature selection in high-dimensional continuous regression"
    ],
    "time_complexity": "O(n * d)",
    "space_complexity": "O(d)",
    "advantages": [
      "Performs feature selection automatically",
      "Produces sparse models"
    ],
    "limitations": [
      "Struggles when features are highly correlated"
    ],
    "alternatives": [
      "Elastic Net Regression",
      "Ridge Regression"
    ],
    "python_template": "from sklearn.linear_model import Lasso\nmodel = Lasso(alpha=0.1)\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "elastic_net_regression",
    "name": "Elastic Net Regression",
    "category": "Regression",
    "description": "Linear regression combining L1 and L2 regularizations to balance feature selection and coefficient stability.",
    "problem_patterns": [
      "combined l1 l2 regression",
      "regularized correlated feature selection"
    ],
    "data_structures": [
      "Coefficient Vector"
    ],
    "requirements": [
      "Scaled numeric features"
    ],
    "input_characteristics": [
      "High-dimensional tabular data with correlated feature groups"
    ],
    "constraints": [
      "Requires tuning two parameters (alpha, l1_ratio)"
    ],
    "best_use_cases": [
      "Continuous target prediction with group-correlated high-dimensional features"
    ],
    "time_complexity": "O(n * d)",
    "space_complexity": "O(d)",
    "advantages": [
      "Combines benefits of Lasso and Ridge",
      "Handles correlated feature groups"
    ],
    "limitations": [
      "More parameters to tune"
    ],
    "alternatives": [
      "Lasso Regression",
      "Ridge Regression"
    ],
    "python_template": "from sklearn.linear_model import ElasticNet\nmodel = ElasticNet(alpha=0.1, l1_ratio=0.5)\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "polynomial_regression",
    "name": "Polynomial Regression",
    "category": "Regression",
    "description": "Linear model variant incorporating polynomial feature combinations to fit non-linear curves.",
    "problem_patterns": [
      "curved trend fitting",
      "polynomial feature regression",
      "non linear continuous relationship"
    ],
    "data_structures": [
      "Polynomial Feature Matrix"
    ],
    "requirements": [
      "Continuous numeric input"
    ],
    "input_characteristics": [
      "Non-linear continuous relationships"
    ],
    "constraints": [
      "Feature space grows exponentially with polynomial degree"
    ],
    "best_use_cases": [
      "Fitting non-linear curves for small feature sets"
    ],
    "time_complexity": "O(n * d^degree)",
    "space_complexity": "O(d^degree)",
    "advantages": [
      "Models curved relationships simply"
    ],
    "limitations": [
      "Prone to severe overfitting at higher degrees"
    ],
    "alternatives": [
      "Decision Tree Regressor",
      "SVR"
    ],
    "python_template": "from sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.linear_model import LinearRegression\npoly = PolynomialFeatures(degree=2)\nX_poly = poly.fit_transform(X_train)\nmodel = LinearRegression().fit(X_poly, y_train)"
  },
  {
    "id": "decision_tree_regressor",
    "name": "Decision Tree Regressor",
    "category": "Regression",
    "description": "Non-parametric regression model predicting continuous target by splitting feature space into hyper-rectangles.",
    "problem_patterns": [
      "interpretable continuous rule tree",
      "non linear regression tree"
    ],
    "data_structures": [
      "Decision Tree"
    ],
    "requirements": [
      "Tabular feature matrix"
    ],
    "input_characteristics": [
      "Tabular features with step-like non-linear target patterns"
    ],
    "constraints": [
      "Cannot extrapolate beyond training target range"
    ],
    "best_use_cases": [
      "Non-linear tabular regression with step boundaries"
    ],
    "time_complexity": "O(d * n log n)",
    "space_complexity": "O(nodes)",
    "advantages": [
      "Non-linear capability",
      "Interpretable splits"
    ],
    "limitations": [
      "Predicts constant values per leaf region"
    ],
    "alternatives": [
      "Random Forest Regressor",
      "Gradient Boosting Regressor"
    ],
    "python_template": "from sklearn.tree import DecisionTreeRegressor\nmodel = DecisionTreeRegressor(max_depth=5)\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "random_forest_regressor",
    "name": "Random Forest Regressor",
    "category": "Regression",
    "description": "Ensemble of regression decision trees averaging continuous target predictions.",
    "problem_patterns": [
      "robust continuous prediction",
      "tabular house price estimation",
      "stock/sales forecasting ensemble"
    ],
    "data_structures": [
      "Forest of Regression Trees"
    ],
    "requirements": [
      "Tabular numeric features"
    ],
    "input_characteristics": [
      "Tabular datasets with complex continuous relationships"
    ],
    "constraints": [
      "Cannot extrapolate outside min/max target range"
    ],
    "best_use_cases": [
      "General purpose tabular regression benchmark"
    ],
    "time_complexity": "O(n_trees * d * n log n)",
    "space_complexity": "O(n_trees * nodes)",
    "advantages": [
      "High accuracy",
      "Resistant to overfitting",
      "Provides feature importances"
    ],
    "limitations": [
      "Slower prediction than linear model"
    ],
    "alternatives": [
      "Gradient Boosting Regressor",
      "XGBoost Regressor"
    ],
    "python_template": "from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor(n_estimators=100)\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "extra_trees_regressor",
    "name": "Extra Trees Regressor",
    "category": "Regression",
    "description": "Extremely Randomized Trees regressor randomizing splits for lower variance continuous predictions.",
    "problem_patterns": [
      "extra trees continuous prediction",
      "fast ensemble regressor"
    ],
    "data_structures": [
      "Forest of Decision Trees"
    ],
    "requirements": [
      "Numeric features"
    ],
    "input_characteristics": [
      "Noisy continuous tabular data"
    ],
    "constraints": [
      "Can build large trees"
    ],
    "best_use_cases": [
      "Fast continuous target estimation on noisy datasets"
    ],
    "time_complexity": "O(n_trees * d * n)",
    "space_complexity": "O(n_trees * nodes)",
    "advantages": [
      "Faster than Random Forest",
      "Smoother predictions"
    ],
    "limitations": [
      "Slightly higher bias"
    ],
    "alternatives": [
      "Random Forest Regressor",
      "Gradient Boosting Regressor"
    ],
    "python_template": "from sklearn.ensemble import ExtraTreesRegressor\nmodel = ExtraTreesRegressor(n_estimators=100)\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "gradient_boosting_regressor",
    "name": "Gradient Boosting Regressor",
    "category": "Regression",
    "description": "Sequential boosting model minimizing continuous loss function using decision trees.",
    "problem_patterns": [
      "gradient boosting continuous target",
      "high precision regression ensemble"
    ],
    "data_structures": [
      "Sequence of Regression Trees"
    ],
    "requirements": [
      "Preprocessed features"
    ],
    "input_characteristics": [
      "Structured continuous tabular targets"
    ],
    "constraints": [
      "Requires tuning learning rate and tree depth"
    ],
    "best_use_cases": [
      "High accuracy continuous regression benchmarks"
    ],
    "time_complexity": "O(n_trees * d * n log n)",
    "space_complexity": "O(n_trees * nodes)",
    "advantages": [
      "Top-tier predictive performance"
    ],
    "limitations": [
      "Sensitive to hyper-parameters"
    ],
    "alternatives": [
      "XGBoost Regressor",
      "Random Forest Regressor"
    ],
    "python_template": "from sklearn.ensemble import GradientBoostingRegressor\nmodel = GradientBoostingRegressor()\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "support_vector_regressor",
    "name": "Support Vector Regressor (SVR)",
    "category": "Regression",
    "description": "Regression variant of Support Vector Machines fitting error within an epsilon margin boundary.",
    "problem_patterns": [
      "epsilon margin regression",
      "kernel SVR continuous estimation"
    ],
    "data_structures": [
      "Support Vectors"
    ],
    "requirements": [
      "Feature scaling essential"
    ],
    "input_characteristics": [
      "Small to medium non-linear datasets"
    ],
    "constraints": [
      "Poor scalability for large n"
    ],
    "best_use_cases": [
      "Non-linear regression on small high-dimensional datasets"
    ],
    "time_complexity": "O(n^2 * d) to O(n^3)",
    "space_complexity": "O(n_support_vectors * d)",
    "advantages": [
      "Robust to outliers within margin epsilon"
    ],
    "limitations": [
      "Slow on large datasets"
    ],
    "alternatives": [
      "Random Forest Regressor",
      "Ridge Regression"
    ],
    "python_template": "from sklearn.svm import SVR\nmodel = SVR(kernel='rbf')\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "xgboost_regressor",
    "name": "XGBoost Regressor",
    "category": "Regression",
    "description": "Optimized gradient boosted decision tree implementation tailored for continuous regression problems.",
    "problem_patterns": [
      "xgboost continuous forecasting",
      "competition winning regressor"
    ],
    "data_structures": [
      "DMatrix",
      "Gradient Trees"
    ],
    "requirements": [
      "Numeric features"
    ],
    "input_characteristics": [
      "Structured tabular data with continuous target"
    ],
    "constraints": [
      "Requires tuning hyper-parameters"
    ],
    "best_use_cases": [
      "State-of-the-art tabular continuous regression"
    ],
    "time_complexity": "O(n_trees * d * n log n)",
    "space_complexity": "O(n_trees * nodes)",
    "advantages": [
      "High performance",
      "Handles missing data"
    ],
    "limitations": [
      "Black-box model"
    ],
    "alternatives": [
      "Gradient Boosting Regressor",
      "LightGBM Regressor"
    ],
    "python_template": "import xgboost as xgb\nmodel = xgb.XGBRegressor()\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "bfs_graph",
    "name": "Breadth-First Search (BFS Graph)",
    "category": "Graph Algorithms",
    "description": "Graph traversal algorithm that explores neighbor nodes level by level starting from source node.",
    "problem_patterns": [
      "shortest path unweighted graph",
      "level order graph traversal",
      "social network degrees of separation"
    ],
    "data_structures": [
      "Queue",
      "Adjacency List",
      "Visited Set"
    ],
    "requirements": [
      "Graph adjacency representation"
    ],
    "input_characteristics": [
      "Unweighted graph or tree"
    ],
    "constraints": [
      "O(V) memory for queue"
    ],
    "best_use_cases": [
      "Shortest path in unweighted networks",
      "Finding connected components"
    ],
    "time_complexity": "O(V + E)",
    "space_complexity": "O(V)",
    "advantages": [
      "Guarantees shortest path for unweighted graphs"
    ],
    "limitations": [
      "Does not work for weighted graphs"
    ],
    "alternatives": [
      "Dijkstra's Algorithm",
      "DFS Graph"
    ],
    "python_template": "from collections import deque\ndef bfs_graph(graph, start):\n    visited = {start}; queue = deque([start]); path = []\n    while queue:\n        node = queue.popleft(); path.append(node)\n        for neighbor in graph.get(node, []):\n            if neighbor not in visited:\n                visited.add(neighbor); queue.append(neighbor)\n    return path"
  },
  {
    "id": "dfs_graph",
    "name": "Depth-First Search (DFS Graph)",
    "category": "Graph Algorithms",
    "description": "Graph traversal algorithm that explores as deep as possible along each branch before backtracking.",
    "problem_patterns": [
      "topological ordering",
      "cycle detection in graph",
      "connected components",
      "path existence in graph"
    ],
    "data_structures": [
      "Stack",
      "Recursion",
      "Adjacency List"
    ],
    "requirements": [
      "Graph adjacency representation"
    ],
    "input_characteristics": [
      "Directed or undirected graph"
    ],
    "constraints": [
      "Recursion limit on deep graphs"
    ],
    "best_use_cases": [
      "Detecting graph cycles",
      "Topological sorting",
      "Finding strongly connected components"
    ],
    "time_complexity": "O(V + E)",
    "space_complexity": "O(V)",
    "advantages": [
      "Uses less memory than BFS on deep graphs"
    ],
    "limitations": [
      "Does not find shortest path"
    ],
    "alternatives": [
      "BFS Graph",
      "Tarjan's Algorithm"
    ],
    "python_template": "def dfs_graph(graph, node, visited=None):\n    if visited is None: visited = set()\n    visited.add(node)\n    for neighbor in graph.get(node, []):\n        if neighbor not in visited: dfs_graph(graph, neighbor, visited)\n    return visited"
  },
  {
    "id": "dijkstra",
    "name": "Dijkstra's Shortest Path Algorithm",
    "category": "Graph Algorithms",
    "description": "Graph algorithm that solves single-source shortest path problem for graphs with non-negative edge weights.",
    "problem_patterns": [
      "shortest path weighted graph",
      "minimum distance between cities",
      "network routing shortest distance",
      "gps navigation path"
    ],
    "data_structures": [
      "Priority Queue / Min-Heap",
      "Adjacency List"
    ],
    "requirements": [
      "Non-negative edge weights"
    ],
    "input_characteristics": [
      "Weighted graph with non-negative weights"
    ],
    "constraints": [
      "Fails if graph contains negative edge weights"
    ],
    "best_use_cases": [
      "GPS map routing",
      "Network packet routing"
    ],
    "time_complexity": "O((V + E) log V)",
    "space_complexity": "O(V)",
    "advantages": [
      "Optimal single-source shortest path algorithm"
    ],
    "limitations": [
      "Cannot handle negative edge weights"
    ],
    "alternatives": [
      "Bellman-Ford Algorithm",
      "A* Search"
    ],
    "python_template": "import heapq\ndef dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    pq = [(0, start)]\n    while pq:\n        curr_dist, u = heapq.heappop(pq)\n        if curr_dist > distances[u]: continue\n        for v, weight in graph[u].items():\n            dist = curr_dist + weight\n            if dist < distances[v]:\n                distances[v] = dist; heapq.heappush(pq, (dist, v))\n    return distances"
  },
  {
    "id": "bellman_ford",
    "name": "Bellman-Ford Algorithm",
    "category": "Graph Algorithms",
    "description": "Computes shortest paths from single source to all vertices in a weighted graph, capable of handling negative edge weights.",
    "problem_patterns": [
      "negative edge weight shortest path",
      "detect negative cycle graph",
      "currency arbitrage detection"
    ],
    "data_structures": [
      "Edge List",
      "Distance Array"
    ],
    "requirements": [
      "Edge list representation"
    ],
    "input_characteristics": [
      "Weighted graph with potential negative edge weights"
    ],
    "constraints": [
      "Slower O(V*E) time complexity"
    ],
    "best_use_cases": [
      "Financial currency arbitrage detection",
      "Graphs with negative weights"
    ],
    "time_complexity": "O(V * E)",
    "space_complexity": "O(V)",
    "advantages": [
      "Handles negative edge weights",
      "Detects negative cycles"
    ],
    "limitations": [
      "Slower than Dijkstra's O((V+E) log V)"
    ],
    "alternatives": [
      "Dijkstra's Algorithm",
      "Floyd-Warshall Algorithm"
    ],
    "python_template": "def bellman_ford(edges, V, start):\n    dist = [float('inf')] * V; dist[start] = 0\n    for _ in range(V - 1):\n        for u, v, w in edges:\n            if dist[u] != float('inf') and dist[u] + w < dist[v]: dist[v] = dist[u] + w\n    for u, v, w in edges:\n        if dist[u] != float('inf') and dist[u] + w < dist[v]: return None  # Negative cycle\n    return dist"
  },
  {
    "id": "floyd_warshall",
    "name": "Floyd-Warshall Algorithm",
    "category": "Graph Algorithms",
    "description": "All-pairs shortest path algorithm finding shortest distances between every pair of vertices in a weighted graph.",
    "problem_patterns": [
      "all pairs shortest path",
      "transitive closure matrix",
      "distance matrix all nodes"
    ],
    "data_structures": [
      "2D Matrix"
    ],
    "requirements": [
      "Graph distance matrix"
    ],
    "input_characteristics": [
      "Dense small to medium graph"
    ],
    "constraints": [
      "O(V^3) time makes it unsuitable for large V (> 1000)"
    ],
    "best_use_cases": [
      "Precomputing all-pairs distances for small dense graphs"
    ],
    "time_complexity": "O(V^3)",
    "space_complexity": "O(V^2)",
    "advantages": [
      "Simple 3-loop implementation",
      "Computes all-pairs distances at once"
    ],
    "limitations": [
      "Cubic time complexity O(V^3)"
    ],
    "alternatives": [
      "Johnson's Algorithm",
      "Repeated Dijkstra"
    ],
    "python_template": "def floyd_warshall(V, dist):\n    for k in range(V):\n        for i in range(V):\n            for j in range(V):\n                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])\n    return dist"
  },
  {
    "id": "a_star_search",
    "name": "A* Search Algorithm",
    "category": "Graph Algorithms",
    "description": "Heuristic-driven graph search algorithm that finds the shortest path using f(n) = g(n) + h(n).",
    "problem_patterns": [
      "heuristic shortest path",
      "game pathfinding",
      "maze solver with heuristic"
    ],
    "data_structures": [
      "Priority Queue",
      "Heuristic Function"
    ],
    "requirements": [
      "Admissible heuristic function (h(n) <= actual cost)"
    ],
    "input_characteristics": [
      "Grid map or spatial graph with coordinate data"
    ],
    "constraints": [
      "Quality depends heavily on heuristic function"
    ],
    "best_use_cases": [
      "Video game NPC pathfinding",
      "Robotic navigation"
    ],
    "time_complexity": "O(E) worst case",
    "space_complexity": "O(V)",
    "advantages": [
      "Much faster than Dijkstra when good heuristic is available"
    ],
    "limitations": [
      "Memory usage for storing open set"
    ],
    "alternatives": [
      "Dijkstra's Algorithm",
      "BFS Graph"
    ],
    "python_template": "import heapq\ndef a_star(graph, start, goal, heuristic):\n    open_set = [(0 + heuristic(start, goal), 0, start)]; g_score = {start: 0}\n    while open_set:\n        f, g, current = heapq.heappop(open_set)\n        if current == goal: return g\n        for neighbor, weight in graph.get(current, []):\n            tentative_g = g + weight\n            if tentative_g < g_score.get(neighbor, float('inf')):\n                g_score[neighbor] = tentative_g\n                heapq.heappush(open_set, (tentative_g + heuristic(neighbor, goal), tentative_g, neighbor))\n    return float('inf')"
  },
  {
    "id": "prim_algorithm",
    "name": "Prim's Algorithm",
    "category": "Graph Algorithms",
    "description": "Greedy algorithm that finds a Minimum Spanning Tree (MST) for a weighted undirected graph by expanding tree from source.",
    "problem_patterns": [
      "minimum spanning tree prim",
      "connect all nodes with minimum cost",
      "network cable layout minimum weight"
    ],
    "data_structures": [
      "Priority Queue",
      "Adjacency List"
    ],
    "requirements": [
      "Connected undirected weighted graph"
    ],
    "input_characteristics": [
      "Dense connected weighted graph"
    ],
    "constraints": [
      "Must be undirected graph"
    ],
    "best_use_cases": [
      "Designing minimal cost electrical grids or fiber network cabling"
    ],
    "time_complexity": "O(E log V)",
    "space_complexity": "O(V + E)",
    "advantages": [
      "Efficient for dense graphs"
    ],
    "limitations": [
      "Requires graph to be connected"
    ],
    "alternatives": [
      "Kruskal's Algorithm",
      "Boruvka's Algorithm"
    ],
    "python_template": "import heapq\ndef prims(V, adj):\n    visited = [False] * V; pq = [(0, 0)]; mst_cost = 0\n    while pq:\n        weight, u = heapq.heappop(pq)\n        if visited[u]: continue\n        visited[u] = True; mst_cost += weight\n        for v, w in adj[u]:\n            if not visited[v]: heapq.heappush(pq, (w, v))\n    return mst_cost"
  },
  {
    "id": "kruskal_algorithm",
    "name": "Kruskal's Algorithm",
    "category": "Graph Algorithms",
    "description": "Greedy algorithm that constructs Minimum Spanning Tree by sorting edges by weight and adding them using Union-Find.",
    "problem_patterns": [
      "minimum spanning tree kruskal",
      "union find mst",
      "forest spanning tree"
    ],
    "data_structures": [
      "Disjoint Set Union (DSU)",
      "Edge List"
    ],
    "requirements": [
      "Weighted undirected graph"
    ],
    "input_characteristics": [
      "Sparse connected or disconnected weighted graph"
    ],
    "constraints": [
      "Sorting edges requires O(E log E)"
    ],
    "best_use_cases": [
      "Minimum Spanning Tree on sparse graphs"
    ],
    "time_complexity": "O(E log E)",
    "space_complexity": "O(V + E)",
    "advantages": [
      "Works naturally on disconnected graphs (spanning forest)"
    ],
    "limitations": [
      "Edge sorting overhead"
    ],
    "alternatives": [
      "Prim's Algorithm"
    ],
    "python_template": "class DSU:\n    def __init__(self, n): self.parent = list(range(n))\n    def find(self, i):\n        if self.parent[i] == i: return i\n        self.parent[i] = self.find(self.parent[i]); return self.parent[i]\n    def union(self, i, j):\n        root_i, root_j = self.find(i), self.find(j)\n        if root_i != root_j: self.parent[root_i] = root_j; return True\n        return False\ndef kruskal(V, edges):\n    edges.sort(key=lambda x: x[2]); dsu = DSU(V); mst_cost = 0\n    for u, v, w in edges:\n        if dsu.union(u, v): mst_cost += w\n    return mst_cost"
  },
  {
    "id": "topological_sort",
    "name": "Topological Sort",
    "category": "Graph Algorithms",
    "description": "Linear ordering of vertices in a Directed Acyclic Graph (DAG) such that for every directed edge u -> v, u comes before v.",
    "problem_patterns": [
      "dependency resolution order",
      "task scheduling DAG",
      "build order compilation",
      "prerequisite course ordering"
    ],
    "data_structures": [
      "Queue (Kahn's)",
      "In-degree Array",
      "DAG"
    ],
    "requirements": [
      "Directed Acyclic Graph (DAG)"
    ],
    "input_characteristics": [
      "Directed graph with no cycles"
    ],
    "constraints": [
      "Fails if graph contains cycles"
    ],
    "best_use_cases": [
      "Package dependency management (e.g. pip, npm)",
      "Task execution scheduling"
    ],
    "time_complexity": "O(V + E)",
    "space_complexity": "O(V)",
    "advantages": [
      "Detects cycles while computing valid execution sequence"
    ],
    "limitations": [
      "Only applies to DAGs"
    ],
    "alternatives": [
      "DFS Topological Sort"
    ],
    "python_template": "from collections import deque\ndef topological_sort(V, adj):\n    in_degree = [0] * V\n    for u in range(V):\n        for v in adj[u]: in_degree[v] += 1\n    queue = deque([i for i in range(V) if in_degree[i] == 0]); topo = []\n    while queue:\n        u = queue.popleft(); topo.append(u)\n        for v in adj[u]:\n            in_degree[v] -= 1\n            if in_degree[v] == 0: queue.append(v)\n    return topo if len(topo) == V else []"
  },
  {
    "id": "union_find",
    "name": "Union-Find (Disjoint Set Union)",
    "category": "Graph Algorithms",
    "description": "Data structure tracking a partition of a set into disjoint subsets, supporting union and find with path compression.",
    "problem_patterns": [
      "connected components tracking",
      "dynamic connectivity",
      "redundant connection detection"
    ],
    "data_structures": [
      "Parent Array",
      "Rank / Size Array"
    ],
    "requirements": [
      "Element index bounds"
    ],
    "input_characteristics": [
      "Dynamic edge additions"
    ],
    "constraints": [
      "Does not easily support edge deletions"
    ],
    "best_use_cases": [
      "Cycle detection in undirected graphs",
      "Kruskal's MST",
      "Image connected component labeling"
    ],
    "time_complexity": "O(alpha(N)) per operation",
    "space_complexity": "O(N)",
    "advantages": [
      "Near constant time operations O(alpha(N))"
    ],
    "limitations": [
      "Cannot easily delete set connections"
    ],
    "alternatives": [
      "DFS Connected Components"
    ],
    "python_template": "class DSU:\n    def __init__(self, n):\n        self.p = list(range(n))\n    def find(self, x):\n        if self.p[x] != x: self.p[x] = self.find(self.p[x])\n        return self.p[x]\n    def union(self, x, y):\n        rx, ry = self.find(x), self.find(y)\n        if rx != ry: self.p[rx] = ry; return True\n        return False"
  },
  {
    "id": "johnson_algorithm",
    "name": "Johnson's Algorithm",
    "category": "Graph Algorithms",
    "description": "All-pairs shortest path algorithm combining Bellman-Ford and Dijkstra to handle sparse graphs with negative edge weights.",
    "problem_patterns": [
      "sparse graph all pairs shortest path",
      "negative edge weight all pairs"
    ],
    "data_structures": [
      "Min-Heap",
      "Adjacency List"
    ],
    "requirements": [
      "No negative cycles"
    ],
    "input_characteristics": [
      "Sparse weighted graph with potential negative edge weights"
    ],
    "constraints": [
      "Complex multi-stage re-weighting logic"
    ],
    "best_use_cases": [
      "All-pairs shortest path on sparse graphs with negative edge weights"
    ],
    "time_complexity": "O(V^2 log V + V * E)",
    "space_complexity": "O(V + E)",
    "advantages": [
      "Faster than Floyd-Warshall for sparse graphs"
    ],
    "limitations": [
      "Higher implementation complexity"
    ],
    "alternatives": [
      "Floyd-Warshall Algorithm"
    ],
    "python_template": "# Johnson's algorithm reweights edges using Bellman-Ford, then runs Dijkstra from each node."
  },
  {
    "id": "binary_search_tree",
    "name": "Binary Search Tree (BST)",
    "category": "Tree Algorithms",
    "description": "Node-based binary tree data structure where left child is smaller and right child is greater than parent.",
    "problem_patterns": [
      "ordered key insertion lookup",
      "dynamic sorted set",
      "range query BST"
    ],
    "data_structures": [
      "Binary Tree Nodes"
    ],
    "requirements": [
      "Comparable key values"
    ],
    "input_characteristics": [
      "Dynamic stream of comparable elements"
    ],
    "constraints": [
      "Can degenerate to O(n) linked list if unbalanced"
    ],
    "best_use_cases": [
      "Dynamic set tracking with dynamic insertion/deletion"
    ],
    "time_complexity": "O(log n) avg, O(n) worst",
    "space_complexity": "O(n)",
    "advantages": [
      "In-order traversal produces sorted keys"
    ],
    "limitations": [
      "Unbalanced trees degrade performance"
    ],
    "alternatives": [
      "AVL Tree",
      "Red-Black Tree"
    ],
    "python_template": "class TreeNode:\n    def __init__(self, val):\n        self.val = val; self.left = None; self.right = None\ndef bst_insert(root, val):\n    if not root: return TreeNode(val)\n    if val < root.val: root.left = bst_insert(root.left, val)\n    else: root.right = bst_insert(root.right, val)\n    return root"
  },
  {
    "id": "avl_tree",
    "name": "AVL Tree",
    "category": "Tree Algorithms",
    "description": "Self-balancing binary search tree where height difference between left and right subtrees is at most one.",
    "problem_patterns": [
      "guaranteed balanced BST",
      "strict O(log n) tree search",
      "frequent lookup balanced tree"
    ],
    "data_structures": [
      "Balanced Binary Tree Nodes"
    ],
    "requirements": [
      "Height balance maintenance"
    ],
    "input_characteristics": [
      "Dynamic ordered keys requiring fast lookups"
    ],
    "constraints": [
      "More rotations during insertions/deletions than Red-Black tree"
    ],
    "best_use_cases": [
      "Lookup-heavy databases requiring strict height balance"
    ],
    "time_complexity": "O(log n) search/insert/delete",
    "space_complexity": "O(n)",
    "advantages": [
      "Guaranteed O(log n) operation bounds"
    ],
    "limitations": [
      "Frequent rebalancing rotation overhead on insertion"
    ],
    "alternatives": [
      "Red-Black Tree",
      "B-Tree"
    ],
    "python_template": "# Self-balancing binary tree utilizing LL, RR, LR, RL rotations to maintain height balance."
  },
  {
    "id": "red_black_tree",
    "name": "Red-Black Tree",
    "category": "Tree Algorithms",
    "description": "Self-balancing binary search tree using node color attributes to ensure log(n) height balance.",
    "problem_patterns": [
      "standard map set tree balance",
      "balanced tree insert heavy"
    ],
    "data_structures": [
      "Red/Black Binary Nodes"
    ],
    "requirements": [
      "Node coloring rules"
    ],
    "input_characteristics": [
      "Frequent dynamic insertions and deletions"
    ],
    "constraints": [
      "Complex node recoloring/rotation rules"
    ],
    "best_use_cases": [
      "Underlying implementation for C++ std::map, Java TreeMap"
    ],
    "time_complexity": "O(log n)",
    "space_complexity": "O(n)",
    "advantages": [
      "Faster insertions/deletions than AVL trees"
    ],
    "limitations": [
      "Slightly taller tree height than AVL"
    ],
    "alternatives": [
      "AVL Tree",
      "B-Tree"
    ],
    "python_template": "# Self-balancing BST using color flags (Red/Black) for tree property maintenance."
  },
  {
    "id": "binary_tree_traversal",
    "name": "Binary Tree Traversal (Pre/In/Post Order)",
    "category": "Tree Algorithms",
    "description": "Depth-first recursive or iterative methods to visit every node in a binary tree in structured order.",
    "problem_patterns": [
      "tree traversal pre order in order post order",
      "serialize deserialize tree",
      "evaluate expression tree"
    ],
    "data_structures": [
      "Binary Tree",
      "Stack"
    ],
    "requirements": [
      "Binary tree structure"
    ],
    "input_characteristics": [
      "Hierarchical tree structure"
    ],
    "constraints": [
      "Recursion stack depth"
    ],
    "best_use_cases": [
      "In-order for sorted BST output",
      "Post-order for tree deletion/expression evaluation"
    ],
    "time_complexity": "O(n)",
    "space_complexity": "O(h)",
    "advantages": [
      "Simple recursive implementations"
    ],
    "limitations": [
      "Call stack overhead"
    ],
    "alternatives": [
      "Level Order Traversal"
    ],
    "python_template": "def inorder(root):\n    return inorder(root.left) + [root.val] + inorder(root.right) if root else []"
  },
  {
    "id": "tree_level_order",
    "name": "Tree Level Order Traversal (BFS Tree)",
    "category": "Tree Algorithms",
    "description": "Breadth-first traversal visiting binary tree nodes level by level from top to bottom.",
    "problem_patterns": [
      "level order binary tree",
      "find bottom left node",
      "zigzag level traversal"
    ],
    "data_structures": [
      "Queue",
      "Binary Tree"
    ],
    "requirements": [
      "Tree node structure"
    ],
    "input_characteristics": [
      "Tree hierarchical levels"
    ],
    "constraints": [
      "Memory proportional to maximum tree width"
    ],
    "best_use_cases": [
      "Printing tree by levels",
      "Finding shallowest leaf node"
    ],
    "time_complexity": "O(n)",
    "space_complexity": "O(w) maximum width",
    "advantages": [
      "Processes nodes in natural top-down depth order"
    ],
    "limitations": [
      "Queue size for wide trees"
    ],
    "alternatives": [
      "DFS Tree Traversal"
    ],
    "python_template": "from collections import deque\ndef level_order(root):\n    if not root: return []\n    q = deque([root]); res = []\n    while q:\n        level = []\n        for _ in range(len(q)):\n            node = q.popleft(); level.append(node.val)\n            if node.left: q.append(node.left)\n            if node.right: q.append(node.right)\n        res.append(level)\n    return res"
  },
  {
    "id": "trie_tree",
    "name": "Trie (Prefix Tree)",
    "category": "Tree Algorithms",
    "description": "Tree-like data structure used to store a dynamic set or associative array where keys are usually strings.",
    "problem_patterns": [
      "autocomplete string prefix",
      "dictionary word lookup",
      "prefix matching",
      "spell checker trie"
    ],
    "data_structures": [
      "Trie Node",
      "Character Map"
    ],
    "requirements": [
      "String inputs"
    ],
    "input_characteristics": [
      "Set of text strings or words"
    ],
    "constraints": [
      "Memory intensive for large alphabets"
    ],
    "best_use_cases": [
      "Search engine autocomplete suggestions",
      "IP routing prefix lookup"
    ],
    "time_complexity": "O(L) per search/insert (L=string length)",
    "space_complexity": "O(ALPHABET_SIZE * N * L)",
    "advantages": [
      "Lookup speed depends only on word length O(L), independent of total dictionary size"
    ],
    "limitations": [
      "High memory consumption"
    ],
    "alternatives": [
      "Hash Table Lookup",
      "Ternary Search Tree"
    ],
    "python_template": "class TrieNode:\n    def __init__(self):\n        self.children = {}; self.is_end = False\nclass Trie:\n    def __init__(self): self.root = TrieNode()\n    def insert(self, word):\n        curr = self.root\n        for ch in word:\n            if ch not in curr.children: curr.children[ch] = TrieNode()\n            curr = curr.children[ch]\n        curr.is_end = True\n    def starts_with(self, prefix):\n        curr = self.root\n        for ch in prefix:\n            if ch not in curr.children: return False\n            curr = curr.children[ch]\n        return True"
  },
  {
    "id": "binary_heap",
    "name": "Binary Heap (Priority Queue)",
    "category": "Tree Algorithms",
    "description": "Complete binary tree satisfying heap property (min-heap or max-heap), enabling O(1) access to min/max element.",
    "problem_patterns": [
      "priority queue heap",
      "top k frequent elements",
      "merge k sorted lists",
      "find median in data stream"
    ],
    "data_structures": [
      "Array representation of Complete Tree"
    ],
    "requirements": [
      "Comparable items"
    ],
    "input_characteristics": [
      "Dynamic stream requiring min/max retrieval"
    ],
    "constraints": [
      "Searching arbitrary elements takes O(n)"
    ],
    "best_use_cases": [
      "Priority queues",
      "Task scheduling",
      "Top K element tracking"
    ],
    "time_complexity": "O(1) peek, O(log n) push/pop",
    "space_complexity": "O(n)",
    "advantages": [
      "Constant time O(1) access to min/max"
    ],
    "limitations": [
      "Arbitrary search is linear O(n)"
    ],
    "alternatives": [
      "Fibonacci Heap",
      "Balanced BST"
    ],
    "python_template": "import heapq\nhp = []\nheapq.heappush(hp, 10)\nmin_val = heapq.heappop(hp)"
  },
  {
    "id": "segment_tree",
    "name": "Segment Tree",
    "category": "Tree Algorithms",
    "description": "Tree data structure for storing intervals or segments, allowing querying range sum/min/max in O(log n).",
    "problem_patterns": [
      "range sum query with update",
      "range minimum query dynamic",
      "point update range query"
    ],
    "data_structures": [
      "Array Segment Tree"
    ],
    "requirements": [
      "Associative operator (sum, min, max, gcd)"
    ],
    "input_characteristics": [
      "Array with frequent point updates and range queries"
    ],
    "constraints": [
      "Requires 4N memory space"
    ],
    "best_use_cases": [
      "Dynamic range minimum/maximum/sum queries with point updates"
    ],
    "time_complexity": "O(log n) query/update, O(n) build",
    "space_complexity": "O(n)",
    "advantages": [
      "Fast O(log n) range query and point update"
    ],
    "limitations": [
      "Complex implementation"
    ],
    "alternatives": [
      "Fenwick Tree (Binary Indexed Tree)",
      "Sparse Table"
    ],
    "python_template": "class SegmentTree:\n    def __init__(self, arr):\n        self.n = len(arr); self.tree = [0] * (4 * self.n)\n        if self.n > 0: self._build(arr, 0, 0, self.n - 1)\n    def _build(self, arr, node, l, r):\n        if l == r: self.tree[node] = arr[l]; return\n        mid = (l + r) // 2\n        self._build(arr, 2*node+1, l, mid); self._build(arr, 2*node+2, mid+1, r)\n        self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]"
  },
  {
    "id": "fenwick_tree",
    "name": "Fenwick Tree (Binary Indexed Tree)",
    "category": "Tree Algorithms",
    "description": "Data structure providing efficient O(log n) methods for prefix sums and point updates on arrays.",
    "problem_patterns": [
      "prefix sum update",
      "binary indexed tree prefix sum",
      "count smaller numbers after self"
    ],
    "data_structures": [
      "Array (1-indexed)"
    ],
    "requirements": [
      "Invertible binary operation (addition)"
    ],
    "input_characteristics": [
      "1D numerical array with point updates"
    ],
    "constraints": [
      "Difficult for arbitrary non-invertible range operations"
    ],
    "best_use_cases": [
      "Dynamic prefix sum queries with point updates"
    ],
    "time_complexity": "O(log n) update/query",
    "space_complexity": "O(n)",
    "advantages": [
      "Easier to code and lower memory overhead than Segment Tree"
    ],
    "limitations": [
      "Limited to prefix operations"
    ],
    "alternatives": [
      "Segment Tree"
    ],
    "python_template": "class FenwickTree:\n    def __init__(self, n):\n        self.tree = [0] * (n + 1)\n    def update(self, i, delta):\n        while i < len(self.tree):\n            self.tree[i] += delta; i += i & (-i)\n    def query(self, i):\n        s = 0\n        while i > 0:\n            s += self.tree[i]; i -= i & (-i)\n        return s"
  },
  {
    "id": "lowest_common_ancestor",
    "name": "Lowest Common Ancestor (LCA)",
    "category": "Tree Algorithms",
    "description": "Algorithm finding the deepest shared ancestor node of two target nodes in a tree structure.",
    "problem_patterns": [
      "lowest common ancestor tree",
      "shared ancestor in tree",
      "tree node distance query"
    ],
    "data_structures": [
      "Tree",
      "Binary Lifting Table"
    ],
    "requirements": [
      "Rooted tree structure"
    ],
    "input_characteristics": [
      "Rooted tree with node pairs"
    ],
    "constraints": [
      "Binary lifting requires O(N log N) preprocessing"
    ],
    "best_use_cases": [
      "Genealogy trees",
      "Distance between two nodes in a tree"
    ],
    "time_complexity": "O(log n) query with binary lifting",
    "space_complexity": "O(n log n)",
    "advantages": [
      "Logarithmic query time for repeated queries"
    ],
    "limitations": [
      "Requires tree preprocessing"
    ],
    "alternatives": [
      "RMQ via Euler Tour"
    ],
    "python_template": "def lca_binary_tree(root, p, q):\n    if not root or root == p or root == q: return root\n    left = lca_binary_tree(root.left, p, q); right = lca_binary_tree(root.right, p, q)\n    if left and right: return root\n    return left if left else right"
  },
  {
    "id": "knapsack_01",
    "name": "0/1 Knapsack",
    "category": "Dynamic Programming",
    "description": "Classic DP optimization problem determining maximum value items to select into a fixed capacity knapsack without item duplication.",
    "problem_patterns": [
      "0/1 knapsack problem",
      "subset sum target capacity",
      "equal subset partition",
      "item selection profit capacity"
    ],
    "data_structures": [
      "2D DP Array / 1D DP Array"
    ],
    "requirements": [
      "Discrete integer capacity"
    ],
    "input_characteristics": [
      "Item weights, values, and max capacity W"
    ],
    "constraints": [
      "Pseudo-polynomial time O(N * W)"
    ],
    "best_use_cases": [
      "Resource allocation under strict budget constraints"
    ],
    "time_complexity": "O(N * W)",
    "space_complexity": "O(W)",
    "advantages": [
      "Guarantees globally optimal selection"
    ],
    "limitations": [
      "Infeasible if capacity W is extremely large"
    ],
    "alternatives": [
      "Fractional Knapsack (Greedy)",
      "Unbounded Knapsack"
    ],
    "python_template": "def knapsack_01(weights, values, W):\n    dp = [0] * (W + 1)\n    for w, v in zip(weights, values):\n        for cap in range(W, w - 1, -1):\n            dp[cap] = max(dp[cap], dp[cap - w] + v)\n    return dp[W]"
  },
  {
    "id": "unbounded_knapsack",
    "name": "Unbounded Knapsack",
    "category": "Dynamic Programming",
    "description": "DP problem maximizing item value within capacity when items can be selected infinitely many times.",
    "problem_patterns": [
      "unbounded knapsack unlimited items",
      "rod cutting profit max",
      "coin change maximum total value"
    ],
    "data_structures": [
      "1D DP Array"
    ],
    "requirements": [
      "Integer capacity"
    ],
    "input_characteristics": [
      "Weights, values, capacity W"
    ],
    "constraints": [
      "Pseudo-polynomial complexity"
    ],
    "best_use_cases": [
      "Stock cutter problem",
      "Unlimited item combination maximization"
    ],
    "time_complexity": "O(N * W)",
    "space_complexity": "O(W)",
    "advantages": [
      "Handles unlimited item reuse"
    ],
    "limitations": [
      "Pseudo-polynomial time"
    ],
    "alternatives": [
      "0/1 Knapsack",
      "Coin Change"
    ],
    "python_template": "def unbounded_knapsack(weights, values, W):\n    dp = [0] * (W + 1)\n    for cap in range(1, W + 1):\n        for w, v in zip(weights, values):\n            if w <= cap: dp[cap] = max(dp[cap], dp[cap - w] + v)\n    return dp[W]"
  },
  {
    "id": "coin_change",
    "name": "Coin Change",
    "category": "Dynamic Programming",
    "description": "DP algorithm finding minimum number of coins needed to make up a target amount.",
    "problem_patterns": [
      "coin change minimum coins",
      "minimum items to form target amount",
      "make change target sum"
    ],
    "data_structures": [
      "1D DP Array"
    ],
    "requirements": [
      "Positive coin denominations"
    ],
    "input_characteristics": [
      "List of coin values and integer target amount"
    ],
    "constraints": [
      "Unbounded coin supply assumption"
    ],
    "best_use_cases": [
      "Vending machine currency exchange",
      "Minimum transaction token count"
    ],
    "time_complexity": "O(N * amount)",
    "space_complexity": "O(amount)",
    "advantages": [
      "Optimal minimal count guarantee"
    ],
    "limitations": [
      "Fails to return if amount cannot be formed"
    ],
    "alternatives": [
      "Greedy Change (works for standard canonical coin systems)"
    ],
    "python_template": "def coin_change(coins, amount):\n    dp = [float('inf')] * (amount + 1); dp[0] = 0\n    for coin in coins:\n        for x in range(coin, amount + 1):\n            dp[x] = min(dp[x], dp[x - coin] + 1)\n    return dp[amount] if dp[amount] != float('inf') else -1"
  },
  {
    "id": "longest_common_subsequence",
    "name": "Longest Common Subsequence (LCS)",
    "category": "Dynamic Programming",
    "description": "Finds the longest subsequence present in two sequence strings in the same relative order.",
    "problem_patterns": [
      "longest common subsequence string",
      "diff tool text comparison",
      "dna sequence alignment lcs"
    ],
    "data_structures": [
      "2D DP Grid"
    ],
    "requirements": [
      "Two input sequences/strings"
    ],
    "input_characteristics": [
      "Strings or element sequences"
    ],
    "constraints": [
      "Quadratic space/time O(M * N)"
    ],
    "best_use_cases": [
      "Git diff file comparison",
      "Bioinformatics DNA sequence matching"
    ],
    "time_complexity": "O(M * N)",
    "space_complexity": "O(M * N)",
    "advantages": [
      "Standard exact sequence similarity metric"
    ],
    "limitations": [
      "Quadratic time for long sequences"
    ],
    "alternatives": [
      "Edit Distance",
      "Longest Common Substring"
    ],
    "python_template": "def lcs(text1, text2):\n    m, n = len(text1), len(text2)\n    dp = [[0]*(n+1) for _ in range(m+1)]\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            if text1[i-1] == text2[j-1]: dp[i][j] = dp[i-1][j-1] + 1\n            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[m][n]"
  },
  {
    "id": "longest_increasing_subsequence",
    "name": "Longest Increasing Subsequence (LIS)",
    "category": "Dynamic Programming",
    "description": "Finds length of longest strictly increasing subsequence in an array of numbers.",
    "problem_patterns": [
      "longest increasing subsequence lis",
      "maximum sorted sub-array length",
      "patience sorting"
    ],
    "data_structures": [
      "1D Array / Tails Array"
    ],
    "requirements": [
      "Comparable sequence values"
    ],
    "input_characteristics": [
      "Numerical array"
    ],
    "constraints": [
      "Standard DP O(n^2), Binary Search optimized O(n log n)"
    ],
    "best_use_cases": [
      "Stock price trend analysis",
      "Task ordering optimization"
    ],
    "time_complexity": "O(n log n)",
    "space_complexity": "O(n)",
    "advantages": [
      "Fast O(n log n) via binary search patience sorting"
    ],
    "limitations": [
      "Requires total order on elements"
    ],
    "alternatives": [
      "Patience Sorting"
    ],
    "python_template": "import bisect\ndef lis(nums):\n    tails = []\n    for x in nums:\n        idx = bisect.bisect_left(tails, x)\n        if idx == len(tails): tails.append(x)\n        else: tails[idx] = x\n    return len(tails)"
  },
  {
    "id": "edit_distance",
    "name": "Edit Distance (Levenshtein Distance)",
    "category": "Dynamic Programming",
    "description": "Calculates minimum number of single-character edits (insertions, deletions, substitutions) to transform one string into another.",
    "problem_patterns": [
      "edit distance string spellcheck",
      "levenshtein distance string similarity",
      "typo correction string similarity"
    ],
    "data_structures": [
      "2D DP Grid"
    ],
    "requirements": [
      "String inputs"
    ],
    "input_characteristics": [
      "Text strings"
    ],
    "constraints": [
      "O(M * N) space and time"
    ],
    "best_use_cases": [
      "Auto-correct spell checkers",
      "Fuzzy string search matching"
    ],
    "time_complexity": "O(M * N)",
    "space_complexity": "O(M * N)",
    "advantages": [
      "Comprehensive string similarity measure"
    ],
    "limitations": [
      "Quadratic time bound"
    ],
    "alternatives": [
      "LCS",
      "Hamming Distance"
    ],
    "python_template": "def min_distance(word1, word2):\n    m, n = len(word1), len(word2)\n    dp = [[0]*(n+1) for _ in range(m+1)]\n    for i in range(m+1): dp[i][0] = i\n    for j in range(n+1): dp[0][j] = j\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            if word1[i-1] == word2[j-1]: dp[i][j] = dp[i-1][j-1]\n            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])\n    return dp[m][n]"
  },
  {
    "id": "matrix_chain_multiplication",
    "name": "Matrix Chain Multiplication",
    "category": "Dynamic Programming",
    "description": "DP algorithm determining optimal parenthesization order to minimize scalar multiplications when multiplying a chain of matrices.",
    "problem_patterns": [
      "optimal matrix parenthesization",
      "matrix multiplication order",
      "interval dp matrix chain"
    ],
    "data_structures": [
      "2D DP Table"
    ],
    "requirements": [
      "Matrix dimension array"
    ],
    "input_characteristics": [
      "Sequence of matrix dimensions"
    ],
    "constraints": [
      "Cubic time complexity O(N^3)"
    ],
    "best_use_cases": [
      "Optimizing matrix computation chains in linear algebra libraries"
    ],
    "time_complexity": "O(N^3)",
    "space_complexity": "O(N^2)",
    "advantages": [
      "Drastically reduces scalar multiplications"
    ],
    "limitations": [
      "Cubic time execution"
    ],
    "alternatives": [
      "Greedy Parenthesization (Suboptimal)"
    ],
    "python_template": "def matrix_chain_order(p):\n    n = len(p) - 1; dp = [[0]*n for _ in range(n)]\n    for l in range(2, n + 1):\n        for i in range(n - l + 1):\n            j = i + l - 1; dp[i][j] = float('inf')\n            for k in range(i, j):\n                q = dp[i][k] + dp[k+1][j] + p[i]*p[k+1]*p[j+1]\n                if q < dp[i][j]: dp[i][j] = q\n    return dp[0][n-1]"
  },
  {
    "id": "rod_cutting",
    "name": "Rod Cutting",
    "category": "Dynamic Programming",
    "description": "DP problem determining maximum revenue obtainable by cutting up a rod of length n into pieces with given price table.",
    "problem_patterns": [
      "rod cutting maximum profit",
      "integer segment partition max value"
    ],
    "data_structures": [
      "1D DP Array"
    ],
    "requirements": [
      "Price array for each cut length"
    ],
    "input_characteristics": [
      "Rod length N and price array"
    ],
    "constraints": [
      "O(N^2) dynamic programming table"
    ],
    "best_use_cases": [
      "Industrial material cutting optimization (steel, wood, fabric)"
    ],
    "time_complexity": "O(N^2)",
    "space_complexity": "O(N)",
    "advantages": [
      "Solves optimal manufacturing material allocation"
    ],
    "limitations": [
      "Quadratic time for large rod lengths"
    ],
    "alternatives": [
      "Unbounded Knapsack"
    ],
    "python_template": "def rod_cutting(prices, n):\n    val = [0] * (n + 1)\n    for i in range(1, n + 1):\n        max_val = -1\n        for j in range(i):\n            max_val = max(max_val, prices[j] + val[i - j - 1])\n        val[i] = max_val\n    return val[n]"
  },
  {
    "id": "grid_path_dp",
    "name": "Grid Path Dynamic Programming",
    "category": "Dynamic Programming",
    "description": "Computes number of unique paths or minimum path sum moving top-left to bottom-right in a grid.",
    "problem_patterns": [
      "grid minimum path sum",
      "unique paths grid obstacles",
      "cherry pickup grid dp"
    ],
    "data_structures": [
      "2D Grid / 1D Row DP"
    ],
    "requirements": [
      "Grid movement restricted to right/down"
    ],
    "input_characteristics": [
      "2D Matrix with path costs or obstacles"
    ],
    "constraints": [
      "Restricted movement direction"
    ],
    "best_use_cases": [
      "Robot navigation grid movement",
      "Tile grid shortest path"
    ],
    "time_complexity": "O(M * N)",
    "space_complexity": "O(N)",
    "advantages": [
      "Linear space dynamic programming possible O(N)"
    ],
    "limitations": [
      "Only works for monotonic grid direction moves"
    ],
    "alternatives": [
      "Dijkstra's Algorithm",
      "A* Search"
    ],
    "python_template": "def min_path_sum(grid):\n    m, n = len(grid), len(grid[0])\n    dp = [float('inf')] * (n + 1); dp[1] = 0\n    for r in range(m):\n        for c in range(n):\n            dp[c+1] = min(dp[c+1], dp[c]) + grid[r][c]\n    return dp[n]"
  },
  {
    "id": "interval_dp",
    "name": "Interval Dynamic Programming",
    "category": "Dynamic Programming",
    "description": "DP technique solving problems by breaking down optimal solutions over sub-intervals [i, j].",
    "problem_patterns": [
      "burst balloons interval dp",
      "strange printer interval dp",
      "merge stones interval dp"
    ],
    "data_structures": [
      "2D Interval DP Table"
    ],
    "requirements": [
      "Sub-problem optimal structure over contiguous ranges"
    ],
    "input_characteristics": [
      "Sequential array requiring range merges"
    ],
    "constraints": [
      "O(N^3) time complexity"
    ],
    "best_use_cases": [
      "Game theory range merging (e.g. Balloon Bursting, Stone Merging)"
    ],
    "time_complexity": "O(N^3)",
    "space_complexity": "O(N^2)",
    "advantages": [
      "Solves complex non-overlapping interval combination problems"
    ],
    "limitations": [
      "Cubic computational complexity"
    ],
    "alternatives": [
      "Divide and Conquer"
    ],
    "python_template": "# Solves interval sub-problems by iterating over range lengths len = 2..N and split points k."
  },
  {
    "id": "bitmask_dp",
    "name": "Bitmask Dynamic Programming",
    "category": "Dynamic Programming",
    "description": "DP technique using integer bitwise representation of subsets to solve NP-hard subset selection problems.",
    "problem_patterns": [
      "traveling salesperson bitmask dp",
      "assign tasks subset bitmask",
      "hamiltonian path bitmask"
    ],
    "data_structures": [
      "1D/2D DP Array indexed by bitmask integer"
    ],
    "requirements": [
      "Small element count N <= 20"
    ],
    "input_characteristics": [
      "Small set size requiring state permutation tracking"
    ],
    "constraints": [
      "Exponential space O(2^N) limits N to <= 20"
    ],
    "best_use_cases": [
      "Exact solution for Traveling Salesperson Problem (TSP) for small N"
    ],
    "time_complexity": "O(N^2 * 2^N)",
    "space_complexity": "O(N * 2^N)",
    "advantages": [
      "Much faster than naive factorial O(N!) permutation search"
    ],
    "limitations": [
      "Exponential memory bound for N > 20"
    ],
    "alternatives": [
      "Backtracking",
      "Heuristic Genetic Algorithms"
    ],
    "python_template": "def tsp_bitmask(dist):\n    n = len(dist); memo = {}\n    def total_cost(mask, u):\n        if mask == (1 << n) - 1: return dist[u][0]\n        if (mask, u) in memo: return memo[(mask, u)]\n        ans = float('inf')\n        for v in range(n):\n            if not (mask & (1 << v)):\n                ans = min(ans, dist[u][v] + total_cost(mask | (1 << v), v))\n        memo[(mask, u)] = ans; return ans\n    return total_cost(1, 0)"
  },
  {
    "id": "activity_selection",
    "name": "Activity Selection Problem",
    "category": "Greedy Algorithms",
    "description": "Greedy choice algorithm selecting maximum number of mutually compatible activities sorted by finish time.",
    "problem_patterns": [
      "activity selection finish time sort",
      "interval scheduling greedy",
      "maximum non overlapping meetings"
    ],
    "data_structures": [
      "Array of Intervals"
    ],
    "requirements": [
      "Activity start and finish times"
    ],
    "input_characteristics": [
      "Set of overlapping interval pairs"
    ],
    "constraints": [
      "Requires initial sorting by finish time"
    ],
    "best_use_cases": [
      "Meeting room scheduling",
      "Resource reservation systems"
    ],
    "time_complexity": "O(n log n)",
    "space_complexity": "O(1)",
    "advantages": [
      "Optimal linear scan after sorting"
    ],
    "limitations": [
      "Only maximizes count, not total duration"
    ],
    "alternatives": [
      "Interval Scheduling DP"
    ],
    "python_template": "def max_activities(start, finish):\n    activities = sorted(zip(start, finish), key=lambda x: x[1])\n    selected = [activities[0]]; last_finish = activities[0][1]\n    for s, f in activities[1:]:\n        if s >= last_finish: selected.append((s, f)); last_finish = f\n    return len(selected)"
  },
  {
    "id": "fractional_knapsack",
    "name": "Fractional Knapsack",
    "category": "Greedy Algorithms",
    "description": "Greedy algorithm selecting items by value-to-weight ratio, allowing partial item fractions to maximize knapsack value.",
    "problem_patterns": [
      "fractional knapsack greedy",
      "value weight ratio greedy",
      "divisible item selection"
    ],
    "data_structures": [
      "Array of Items"
    ],
    "requirements": [
      "Items are continuously divisible"
    ],
    "input_characteristics": [
      "Item weights, values, capacity W"
    ],
    "constraints": [
      "Only valid when items can be divided"
    ],
    "best_use_cases": [
      "Bulk commodity trading (gold dust, oil, grain allocation)"
    ],
    "time_complexity": "O(n log n)",
    "space_complexity": "O(1)",
    "advantages": [
      "Guarantees optimal solution in O(n log n)"
    ],
    "limitations": [
      "Does not work for 0/1 non-divisible items"
    ],
    "alternatives": [
      "0/1 Knapsack (DP)"
    ],
    "python_template": "def fractional_knapsack(weights, values, W):\n    items = sorted(zip(weights, values), key=lambda x: x[1]/x[0], reverse=True)\n    total_val = 0.0\n    for w, v in items:\n        if W >= w: W -= w; total_val += v\n        else: total_val += v * (W / w); break\n    return total_val"
  },
  {
    "id": "huffman_coding",
    "name": "Huffman Coding",
    "category": "Greedy Algorithms",
    "description": "Greedy algorithm building optimal prefix codes for lossless data compression based on character frequencies.",
    "problem_patterns": [
      "huffman encoding lossless compression",
      "frequency tree prefix coding",
      "minimum total weighted path length"
    ],
    "data_structures": [
      "Min-Heap",
      "Binary Tree Nodes"
    ],
    "requirements": [
      "Character frequency mapping"
    ],
    "input_characteristics": [
      "Text stream or character frequency table"
    ],
    "constraints": [
      "Requires two-pass encoding or frequency transmission"
    ],
    "best_use_cases": [
      "ZIP file compression",
      "JPEG image encoding",
      "MP3 audio compression"
    ],
    "time_complexity": "O(n log n)",
    "space_complexity": "O(n)",
    "advantages": [
      "Optimal prefix code generator"
    ],
    "limitations": [
      "Code tree must be sent along compressed data"
    ],
    "alternatives": [
      "Arithmetic Coding",
      "Lempel-Ziv (LZ77)"
    ],
    "python_template": "import heapq\ndef huffman_tree(freqs):\n    pq = [[weight, [char, ]] for char, weight in freqs.items()]\n    heapq.heapify(pq)\n    while len(pq) > 1:\n        lo = heapq.heappop(pq); hi = heapq.heappop(pq)\n        for pair in lo[1:]: pair[1] = '0' + pair[1]\n        for pair in hi[1:]: pair[1] = '1' + pair[1]\n        heapq.heappush(pq, [lo[0] + hi[0]] + lo[1:] + hi[1:])\n    return pq[0][1:]"
  },
  {
    "id": "job_sequencing",
    "name": "Job Sequencing with Deadlines",
    "category": "Greedy Algorithms",
    "description": "Greedy strategy scheduling jobs with deadlines and profits to maximize total profit on a single machine.",
    "problem_patterns": [
      "job scheduling profit deadline",
      "deadline scheduling greedy"
    ],
    "data_structures": [
      "Time Slot Array / Disjoint Set Union"
    ],
    "requirements": [
      "Job profit, deadline, duration"
    ],
    "input_characteristics": [
      "List of jobs with profit and deadline"
    ],
    "constraints": [
      "Each job takes 1 unit of time"
    ],
    "best_use_cases": [
      "Task execution scheduling under deadline penalties"
    ],
    "time_complexity": "O(N^2) or O(N log N) with DSU",
    "space_complexity": "O(max_deadline)",
    "advantages": [
      "Maximizes overall commercial profit"
    ],
    "limitations": [
      "Fixed single-unit time slot assumption"
    ],
    "alternatives": [
      "Earliest Deadline First (EDF)"
    ],
    "python_template": "def job_sequencing(jobs):\n    # jobs = [(id, deadline, profit)]\n    jobs.sort(key=lambda x: x[2], reverse=True)\n    max_d = max(j[1] for j in jobs)\n    slots = [-1] * (max_d + 1); total_profit = 0\n    for j_id, deadline, profit in jobs:\n        for t in range(deadline, 0, -1):\n            if slots[t] == -1: slots[t] = j_id; total_profit += profit; break\n    return total_profit"
  },
  {
    "id": "interval_scheduling",
    "name": "Interval Scheduling",
    "category": "Greedy Algorithms",
    "description": "Greedy choice technique picking maximum number of non-overlapping intervals.",
    "problem_patterns": [
      "interval overlap removal",
      "minimum interval removals to eliminate overlap"
    ],
    "data_structures": [
      "Array of Intervals"
    ],
    "requirements": [
      "Interval start/end boundaries"
    ],
    "input_characteristics": [
      "Array of numeric 2D intervals"
    ],
    "constraints": [
      "Greedy finish time sorting strictly required"
    ],
    "best_use_cases": [
      "Conference talk scheduling",
      "Resource slot allocation"
    ],
    "time_complexity": "O(n log n)",
    "space_complexity": "O(1)",
    "advantages": [
      "Proved optimal greedy strategy"
    ],
    "limitations": [
      "Unweighted interval count only"
    ],
    "alternatives": [
      "Interval DP"
    ],
    "python_template": "def erase_overlap_intervals(intervals):\n    intervals.sort(key=lambda x: x[1]); end = float('-inf'); count = 0\n    for inv in intervals:\n        if inv[0] >= end: end = inv[1]\n        else: count += 1\n    return count"
  },
  {
    "id": "jump_game_greedy",
    "name": "Jump Game (Greedy Reachability)",
    "category": "Greedy Algorithms",
    "description": "Greedy evaluation tracking farthest reachable index at each position to determine if array end can be reached.",
    "problem_patterns": [
      "jump game reachable end",
      "minimum jumps to reach end"
    ],
    "data_structures": [
      "Farthest Reach Integer"
    ],
    "requirements": [
      "Non-negative jump capability per index"
    ],
    "input_characteristics": [
      "Array of max jump lengths"
    ],
    "constraints": [
      "Must evaluate incrementally left to right"
    ],
    "best_use_cases": [
      "Reachability analysis in sequential state transitions"
    ],
    "time_complexity": "O(n)",
    "space_complexity": "O(1)",
    "advantages": [
      "Linear time O(n) single pass, O(1) space"
    ],
    "limitations": [
      "Requires 1D forward progress structure"
    ],
    "alternatives": [
      "BFS Search",
      "Dynamic Programming"
    ],
    "python_template": "def can_jump(nums):\n    max_reach = 0\n    for i, jump in enumerate(nums):\n        if i > max_reach: return False\n        max_reach = max(max_reach, i + jump)\n    return True"
  },
  {
    "id": "gas_station_problem",
    "name": "Gas Station Problem",
    "category": "Greedy Algorithms",
    "description": "Greedy evaluation finding starting circuit index to travel around a circle of gas stations.",
    "problem_patterns": [
      "circular circuit gas station",
      "petrol pump circle journey"
    ],
    "data_structures": [
      "Running Surplus Counter"
    ],
    "requirements": [
      "Total gas >= total cost"
    ],
    "input_characteristics": [
      "Gas array and cost array"
    ],
    "constraints": [
      "Requires total gas sum >= total cost sum"
    ],
    "best_use_cases": [
      "Circular vehicle route energy planning"
    ],
    "time_complexity": "O(n)",
    "space_complexity": "O(1)",
    "advantages": [
      "Linear O(n) single pass solution"
    ],
    "limitations": [
      "Single circular route assumption"
    ],
    "alternatives": [
      "Brute Force Simulation"
    ],
    "python_template": "def can_complete_circuit(gas, cost):\n    if sum(gas) < sum(cost): return -1\n    total = 0; start = 0\n    for i in range(len(gas)):\n        total += gas[i] - cost[i]\n        if total < 0: total = 0; start = i + 1\n    return start"
  },
  {
    "id": "kruskal_greedy",
    "name": "Kruskal's MST (Greedy)",
    "category": "Greedy Algorithms",
    "description": "Greedy edge pick strategy building minimum spanning tree by selecting smallest weight edges.",
    "problem_patterns": [
      "kruskal edge pick greedy",
      "greedy minimum edge spanning"
    ],
    "data_structures": [
      "Union-Find",
      "Sorted Edge List"
    ],
    "requirements": [
      "Undirected weighted edges"
    ],
    "input_characteristics": [
      "Edge list with weights"
    ],
    "constraints": [
      "Sorting step requires O(E log E)"
    ],
    "best_use_cases": [
      "Greedy spanning tree connection"
    ],
    "time_complexity": "O(E log E)",
    "space_complexity": "O(V)",
    "advantages": [
      "Simple greedy edge selection principle"
    ],
    "limitations": [
      "Sorting overhead"
    ],
    "alternatives": [
      "Prim's Algorithm"
    ],
    "python_template": "# Sort edges by weight and pick greedily if no cycle is created."
  },
  {
    "id": "prim_greedy",
    "name": "Prim's MST (Greedy)",
    "category": "Greedy Algorithms",
    "description": "Greedy node expansion picking closest vertex outside current MST tree.",
    "problem_patterns": [
      "prim node grow greedy",
      "closest vertex expansion mst"
    ],
    "data_structures": [
      "Min-Heap"
    ],
    "requirements": [
      "Connected graph"
    ],
    "input_characteristics": [
      "Adjacency weighted list"
    ],
    "constraints": [
      "Undirected graph"
    ],
    "best_use_cases": [
      "Dense graph minimum spanning tree"
    ],
    "time_complexity": "O(E log V)",
    "space_complexity": "O(V)",
    "advantages": [
      "Optimal greedy cut strategy"
    ],
    "limitations": [
      "Requires priority queue maintenance"
    ],
    "alternatives": [
      "Kruskal's Algorithm"
    ],
    "python_template": "# Priority queue extracts minimum weight edge connected to current MST."
  },
  {
    "id": "dijkstra_greedy",
    "name": "Dijkstra Shortest Path (Greedy)",
    "category": "Graph Algorithms",
    "description": "Greedy choice algorithm always picking unvisited node with minimum tentative distance.",
    "problem_patterns": [
      "dijkstra minimum distance greedy choice",
      "shortest distance greedy pick"
    ],
    "data_structures": [
      "Min-Heap"
    ],
    "requirements": [
      "Non-negative weights"
    ],
    "input_characteristics": [
      "Weighted directed/undirected graph"
    ],
    "constraints": [
      "Non-negative edge weights"
    ],
    "best_use_cases": [
      "Single source shortest path"
    ],
    "time_complexity": "O((V + E) log V)",
    "space_complexity": "O(V)",
    "advantages": [
      "Greedy choice property guarantees optimal shortest distance"
    ],
    "limitations": [
      "Fails with negative weights"
    ],
    "alternatives": [
      "Bellman-Ford Algorithm"
    ],
    "python_template": "# Greedily pick smallest distance node from heap."
  },
  {
    "id": "mst_strategies",
    "name": "Minimum Spanning Tree Strategy",
    "category": "Greedy Algorithms",
    "description": "General greedy strategy for connecting graph nodes with minimal total edge weight.",
    "problem_patterns": [
      "minimum spanning tree general greedy",
      "network layout optimization"
    ],
    "data_structures": [
      "Graph",
      "Min Heap / Union Find"
    ],
    "requirements": [
      "Weighted undirected graph"
    ],
    "input_characteristics": [
      "Connected graph"
    ],
    "constraints": [
      "No cycles in output tree"
    ],
    "best_use_cases": [
      "Infrastructure network design"
    ],
    "time_complexity": "O(E log V)",
    "space_complexity": "O(V)",
    "advantages": [
      "Optimal total weight guarantee"
    ],
    "limitations": [
      "Only applies to connected undirected graphs"
    ],
    "alternatives": [
      "Steiner Tree (NP-Hard)"
    ],
    "python_template": "# Applies greedy edge or node selection to form Minimum Spanning Tree."
  },
  {
    "id": "n_queens",
    "name": "N-Queens Problem",
    "category": "Backtracking",
    "description": "Backtracking algorithm placing N chess queens on an N x N chessboard so no two queens attack each other.",
    "problem_patterns": [
      "n queens puzzle",
      "chessboard queen placement non attacking",
      "constraint satisfaction board backtracking"
    ],
    "data_structures": [
      "Column / Diagonal Visited Sets"
    ],
    "requirements": [
      "Board dimension N"
    ],
    "input_characteristics": [
      "Board size N"
    ],
    "constraints": [
      "Exponential search space O(N!)"
    ],
    "best_use_cases": [
      "Constraint satisfaction benchmarks",
      "Puzzle solvers"
    ],
    "time_complexity": "O(N!)",
    "space_complexity": "O(N)",
    "advantages": [
      "Prunes invalid board branches early"
    ],
    "limitations": [
      "Exponential time bound limits N <= 20"
    ],
    "alternatives": [
      "Bitmask Backtracking"
    ],
    "python_template": "def solve_n_queens(n):\n    cols, pos_diag, neg_diag = set(), set(), set(); res = []\n    def backtrack(r, board):\n        if r == n: res.append([\"\".join(row) for row in board]); return\n        for c in range(n):\n            if c in cols or (r+c) in pos_diag or (r-c) in neg_diag: continue\n            cols.add(c); pos_diag.add(r+c); neg_diag.add(r-c); board[r][c] = 'Q'\n            backtrack(r + 1, board)\n            cols.remove(c); pos_diag.remove(r+c); neg_diag.remove(r-c); board[r][c] = '.'\n    backtrack(0, [['.']*n for _ in range(n)])\n    return res"
  },
  {
    "id": "sudoku_solver",
    "name": "Sudoku Solver",
    "category": "Backtracking",
    "description": "Backtracking puzzle solver filling 9x9 grid so every row, column, and 3x3 subgrid contains digits 1-9.",
    "problem_patterns": [
      "sudoku solver 9x9",
      "grid constraint satisfaction backtracking"
    ],
    "data_structures": [
      "2D Grid",
      "Row/Col/Box Hash Sets"
    ],
    "requirements": [
      "Valid 9x9 initial board"
    ],
    "input_characteristics": [
      "9x9 grid matrix with empty slots"
    ],
    "constraints": [
      "Combinatorial worst case"
    ],
    "best_use_cases": [
      "Solving Sudoku puzzles automatically"
    ],
    "time_complexity": "O(9^(N_empty))",
    "space_complexity": "O(81)",
    "advantages": [
      "Solves any valid Sudoku grid"
    ],
    "limitations": [
      "Slow on intentionally adversarial empty grids"
    ],
    "alternatives": [
      "Exact Cover Algorithm X (Knuth)"
    ],
    "python_template": "def solve_sudoku(board):\n    def is_valid(r, c, val):\n        for i in range(9):\n            if board[r][i] == val or board[i][c] == val: return False\n            if board[3*(r//3)+i//3][3*(c//3)+i%3] == val: return False\n        return True\n    def solve():\n        for r in range(9):\n            for c in range(9):\n                if board[r][c] == '.':\n                    for ch in '123456789':\n                        if is_valid(r, c, ch):\n                            board[r][c] = ch\n                            if solve(): return True\n                            board[r][c] = '.'\n                    return False\n        return True\n    solve()"
  },
  {
    "id": "rat_in_maze",
    "name": "Rat in a Maze",
    "category": "Backtracking",
    "description": "Backtracking search finding paths for a rat moving from top-left to bottom-right in a maze with blockages.",
    "problem_patterns": [
      "rat in a maze pathfinding",
      "backtracking grid maze traversal",
      "find all paths in maze"
    ],
    "data_structures": [
      "Recursion Stack",
      "Visited Matrix"
    ],
    "requirements": [
      "2D grid matrix with 0/1 blockages"
    ],
    "input_characteristics": [
      "N x N matrix with open and blocked cells"
    ],
    "constraints": [
      "Avoid infinite loops by tracking visited cells"
    ],
    "best_use_cases": [
      "Finding all valid paths through a maze with obstacles"
    ],
    "time_complexity": "O(4^(N^2))",
    "space_complexity": "O(N^2)",
    "advantages": [
      "Finds all possible valid maze paths"
    ],
    "limitations": [
      "Exponential time bound"
    ],
    "alternatives": [
      "BFS Search (for single shortest path)"
    ],
    "python_template": "def solve_maze(maze):\n    n = len(maze); res = []\n    def backtrack(r, c, path, visited):\n        if r == n - 1 and c == n - 1: res.append(path); return\n        visited.add((r, c))\n        for dr, dc, move in [(1,0,'D'), (0,-1,'L'), (0,1,'R'), (-1,0,'U')]:\n            nr, nc = r + dr, c + dc\n            if 0 <= nr < n and 0 <= nc < n and maze[nr][nc] == 1 and (nr, nc) not in visited:\n                backtrack(nr, nc, path + move, visited)\n        visited.remove((r, c))\n    if maze[0][0] == 1: backtrack(0, 0, \"\", set())\n    return res"
  },
  {
    "id": "permutations_gen",
    "name": "Permutations Generation",
    "category": "Backtracking",
    "description": "Generates all possible ordered arrangements (permutations) of a set of elements using backtracking.",
    "problem_patterns": [
      "generate all permutations",
      "order arrangement permutations",
      "string anagram generator"
    ],
    "data_structures": [
      "Recursion Stack",
      "Visited Boolean Array"
    ],
    "requirements": [
      "List of distinct elements"
    ],
    "input_characteristics": [
      "Sequence of N items"
    ],
    "constraints": [
      "Generates N! total outputs"
    ],
    "best_use_cases": [
      "Brute-force testing all ordering permutations for small N"
    ],
    "time_complexity": "O(N * N!)",
    "space_complexity": "O(N)",
    "advantages": [
      "Generates exact complete set of permutations"
    ],
    "limitations": [
      "Infeasible for N > 12 due to N! growth"
    ],
    "alternatives": [
      "Heap's Algorithm for Permutations"
    ],
    "python_template": "def permute(nums):\n    res = []\n    def backtrack(curr, remaining):\n        if not remaining: res.append(curr); return\n        for i in range(len(remaining)):\n            backtrack(curr + [remaining[i]], remaining[:i] + remaining[i+1:])\n    backtrack([], nums); return res"
  },
  {
    "id": "combinations_gen",
    "name": "Combinations Generation",
    "category": "Backtracking",
    "description": "Backtracking algorithm generating all combinations of k items selected from n distinct numbers.",
    "problem_patterns": [
      "generate n choose k combinations",
      "k element subset selection"
    ],
    "data_structures": [
      "Recursion Stack"
    ],
    "requirements": [
      "Integers n and k"
    ],
    "input_characteristics": [
      "Range 1..n and target length k"
    ],
    "constraints": [
      "Produces C(n, k) total combinations"
    ],
    "best_use_cases": [
      "Lottery combination generation",
      "Team selection permutations"
    ],
    "time_complexity": "O(C(n, k))",
    "space_complexity": "O(k)",
    "advantages": [
      "Generates subset combinations without duplicates"
    ],
    "limitations": [
      "Combinatorial growth if k ~ n/2"
    ],
    "alternatives": [
      "Gosper's Hack for Bitwise Combinations"
    ],
    "python_template": "def combine(n, k):\n    res = []\n    def backtrack(start, curr):\n        if len(curr) == k: res.append(list(curr)); return\n        for i in range(start, n + 1):\n            curr.append(i); backtrack(i + 1, curr); curr.pop()\n    backtrack(1, []); return res"
  },
  {
    "id": "subsets_gen",
    "name": "Subsets Generation (Power Set)",
    "category": "Backtracking",
    "description": "Generates all possible subsets (power set) of a given collection of elements.",
    "problem_patterns": [
      "generate power set",
      "all possible subsets generator",
      "boolean subset selection"
    ],
    "data_structures": [
      "Recursion Stack"
    ],
    "requirements": [
      "Set of unique integers"
    ],
    "input_characteristics": [
      "Array of distinct elements"
    ],
    "constraints": [
      "Generates 2^N total subsets"
    ],
    "best_use_cases": [
      "Evaluating all possible item combinations (power set)"
    ],
    "time_complexity": "O(N * 2^N)",
    "space_complexity": "O(N)",
    "advantages": [
      "Complete state space generation"
    ],
    "limitations": [
      "Exponential size 2^N limits N <= 20"
    ],
    "alternatives": [
      "Bit Manipulation Power Set"
    ],
    "python_template": "def subsets(nums):\n    res = []\n    def backtrack(index, curr):\n        res.append(list(curr))\n        for i in range(index, len(nums)):\n            curr.append(nums[i]); backtrack(i + 1, curr); curr.pop()\n    backtrack(0, []); return res"
  },
  {
    "id": "combination_sum",
    "name": "Combination Sum",
    "category": "Backtracking",
    "description": "Finds all unique combinations of candidate numbers where candidate numbers sum to target.",
    "problem_patterns": [
      "combination sum target",
      "subset sum all combinations"
    ],
    "data_structures": [
      "Recursion Stack"
    ],
    "requirements": [
      "Positive target and candidates"
    ],
    "input_characteristics": [
      "Candidate array and target integer"
    ],
    "constraints": [
      "Duplicate combinations must be pruned"
    ],
    "best_use_cases": [
      "Finding exact coin/token combinations summing to target value"
    ],
    "time_complexity": "O(2^target)",
    "space_complexity": "O(target)",
    "advantages": [
      "Handles repeated item choices"
    ],
    "limitations": [
      "Exponential depth for large target"
    ],
    "alternatives": [
      "0/1 Knapsack DP (if only count or max is needed)"
    ],
    "python_template": "def combination_sum(candidates, target):\n    res = []\n    def backtrack(start, curr, current_sum):\n        if current_sum == target: res.append(list(curr)); return\n        if current_sum > target: return\n        for i in range(start, len(candidates)):\n            curr.append(candidates[i])\n            backtrack(i, curr, current_sum + candidates[i])\n            curr.pop()\n    backtrack(0, [], 0); return res"
  },
  {
    "id": "graph_coloring",
    "name": "Graph Coloring Problem",
    "category": "Backtracking",
    "description": "Determines whether a graph can be colored with at most M colors such that no two adjacent vertices share same color.",
    "problem_patterns": [
      "graph m coloring backtracking",
      "map coloring adjacent region",
      "register allocation graph coloring"
    ],
    "data_structures": [
      "Color Assignment Array",
      "Adjacency Matrix"
    ],
    "requirements": [
      "Graph adjacency structure and M colors"
    ],
    "input_characteristics": [
      "Undirected graph and integer M"
    ],
    "constraints": [
      "NP-complete problem"
    ],
    "best_use_cases": [
      "Compiler register allocation",
      "Map geographical coloring",
      "Exam scheduling"
    ],
    "time_complexity": "O(M^V)",
    "space_complexity": "O(V)",
    "advantages": [
      "Provides exact valid coloring assignment"
    ],
    "limitations": [
      "Exponential time bound O(M^V)"
    ],
    "alternatives": [
      "Welsh-Powell Greedy Coloring"
    ],
    "python_template": "def graph_coloring(graph, m, V):\n    color = [0] * V\n    def is_safe(v, c):\n        for i in range(V):\n            if graph[v][i] and color[i] == c: return False\n        return True\n    def solve(v):\n        if v == V: return True\n        for c in range(1, m + 1):\n            if is_safe(v, c):\n                color[v] = c\n                if solve(v + 1): return True\n                color[v] = 0\n        return False\n    return color if solve(0) else None"
  },
  {
    "id": "hamiltonian_path",
    "name": "Hamiltonian Path / Cycle",
    "category": "Backtracking",
    "description": "Determines if a path exists visiting every node in a graph exactly once.",
    "problem_patterns": [
      "visit every node once hamiltonian",
      "hamiltonian path backtracking",
      "knight tour open path"
    ],
    "data_structures": [
      "Visited Array",
      "Path Array"
    ],
    "requirements": [
      "Graph adjacency representation"
    ],
    "input_characteristics": [
      "Directed or undirected graph"
    ],
    "constraints": [
      "NP-complete problem"
    ],
    "best_use_cases": [
      "Route planning requiring visiting all locations exactly once"
    ],
    "time_complexity": "O(N!)",
    "space_complexity": "O(N)",
    "advantages": [
      "Solves exact visiting sequence"
    ],
    "limitations": [
      "Factorial complexity limits N <= 15"
    ],
    "alternatives": [
      "Bitmask DP (O(N^2 * 2^N))"
    ],
    "python_template": "def hamiltonian_path(graph, V):\n    path = []\n    def solve(u):\n        path.append(u)\n        if len(path) == V: return True\n        for v in graph[u]:\n            if v not in path:\n                if solve(v): return True\n        path.pop(); return False\n    for start in range(V):\n        if solve(start): return path\n    return None"
  },
  {
    "id": "word_search_grid",
    "name": "Word Search Grid",
    "category": "Backtracking",
    "description": "Searches for a word in a 2D character grid by moving adjacent horizontally or vertically without reusing cells.",
    "problem_patterns": [
      "word search 2d grid",
      "grid word boggle solver"
    ],
    "data_structures": [
      "2D Grid",
      "Visited Coordinate Set"
    ],
    "requirements": [
      "Character matrix and target word string"
    ],
    "input_characteristics": [
      "2D char array and string query"
    ],
    "constraints": [
      "Cell reuse within same word path forbidden"
    ],
    "best_use_cases": [
      "Boggle word puzzle games",
      "Sub-string grid pattern search"
    ],
    "time_complexity": "O(N * M * 4^L)",
    "space_complexity": "O(L)",
    "advantages": [
      "Fast early pruning when character mismatches"
    ],
    "limitations": [
      "Worst-case 4^L for matching repeated letter grids"
    ],
    "alternatives": [
      "Trie-based Multi-Word Search"
    ],
    "python_template": "def exist(board, word):\n    R, C = len(board), len(board[0])\n    def dfs(r, c, idx):\n        if idx == len(word): return True\n        if r < 0 or r >= R or c < 0 or c >= C or board[r][c] != word[idx]: return False\n        temp = board[r][c]; board[r][c] = '#'\n        res = dfs(r+1,c,idx+1) or dfs(r-1,c,idx+1) or dfs(r,c+1,idx+1) or dfs(r,c-1,idx+1)\n        board[r][c] = temp; return res\n    for r in range(R):\n        for c in range(C):\n            if dfs(r, c, 0): return True\n    return False"
  },
  {
    "id": "knights_tour",
    "name": "Knight's Tour Problem",
    "category": "Backtracking",
    "description": "Backtracking search finding sequence of moves for a chess knight to visit every square on an N x N board exactly once.",
    "problem_patterns": [
      "knights tour chessboard",
      "knight move board coverage"
    ],
    "data_structures": [
      "Board Matrix"
    ],
    "requirements": [
      "Board dimensions"
    ],
    "input_characteristics": [
      "N x N chessboard"
    ],
    "constraints": [
      "Knight move L-shape restrictions"
    ],
    "best_use_cases": [
      "Chess puzzle algorithms",
      "Graph traversal benchmarks"
    ],
    "time_complexity": "O(8^(N^2))",
    "space_complexity": "O(N^2)",
    "advantages": [
      "Exact coverage solver"
    ],
    "limitations": [
      "Requires Warnsdorff's heuristic for board sizes >= 8x8"
    ],
    "alternatives": [
      "Warnsdorff's Heuristic Rule"
    ],
    "python_template": "# Knight's move backtracking using 8 L-shaped displacement vectors (2,1), (1,2), etc."
  },
  {
    "id": "kmeans_clustering",
    "name": "K-Means Clustering",
    "category": "Machine Learning / AI",
    "description": "Unsupervised clustering algorithm partitioning dataset into K distinct clusters based on distance to centroid.",
    "problem_patterns": [
      "unsupervised customer segmentation",
      "k means cluster grouping",
      "image color quantization",
      "cluster tabular data"
    ],
    "data_structures": [
      "Centroids Matrix",
      "Cluster Assignment Vector"
    ],
    "requirements": [
      "Pre-specified number of clusters K",
      "Scaled numeric features"
    ],
    "input_characteristics": [
      "Continuous numeric feature data"
    ],
    "constraints": [
      "Assumes spherical clusters of similar size"
    ],
    "best_use_cases": [
      "Customer persona segmentation",
      "Document clustering",
      "Image compression via color quantization"
    ],
    "time_complexity": "O(n * k * d * iterations)",
    "space_complexity": "O(n * d + k * d)",
    "advantages": [
      "Fast, scalable unsupervised partitioning"
    ],
    "limitations": [
      "Sensitive to initialization (use k-means++) and outliers"
    ],
    "alternatives": [
      "DBSCAN Clustering",
      "Hierarchical Clustering"
    ],
    "python_template": "from sklearn.cluster import KMeans\nkmeans = KMeans(n_clusters=3, random_state=42)\nlabels = kmeans.fit_predict(X)"
  },
  {
    "id": "dbscan_clustering",
    "name": "DBSCAN Clustering",
    "category": "Machine Learning / AI",
    "description": "Density-based spatial clustering algorithm grouping closely packed points and detecting arbitrary shapes and noise outliers.",
    "problem_patterns": [
      "density based clustering",
      "outlier noise detection clustering",
      "arbitrary shape spatial clusters"
    ],
    "data_structures": [
      "KD-Tree / Ball-Tree"
    ],
    "requirements": [
      "Parameters eps (radius) and min_samples"
    ],
    "input_characteristics": [
      "Spatial or tabular feature coordinates"
    ],
    "constraints": [
      "Struggles with varying density clusters"
    ],
    "best_use_cases": [
      "Geospatial spatial data clustering",
      "Anomaly and outlier detection"
    ],
    "time_complexity": "O(n log n) with tree, O(n^2) worst",
    "space_complexity": "O(n)",
    "advantages": [
      "Does not require specifying K clusters beforehand",
      "Identifies noise/outliers naturally"
    ],
    "limitations": [
      "Sensitive to scale and epsilon parameter setting"
    ],
    "alternatives": [
      "K-Means Clustering",
      "HDBSCAN"
    ],
    "python_template": "from sklearn.cluster import DBSCAN\ndbscan = DBSCAN(eps=0.5, min_samples=5)\nlabels = dbscan.fit_predict(X)"
  },
  {
    "id": "hierarchical_clustering",
    "name": "Hierarchical Clustering",
    "category": "Machine Learning / AI",
    "description": "Agglomerative clustering creating a tree (dendrogram) of nested clusters by merging closest pairs iteratively.",
    "problem_patterns": [
      "dendrogram tree clustering",
      "agglomerative hierarchical clustering",
      "nested cluster taxonomy"
    ],
    "data_structures": [
      "Distance Matrix",
      "Dendrogram Tree"
    ],
    "requirements": [
      "Linkage criterion (ward, complete, average)"
    ],
    "input_characteristics": [
      "Small to medium feature matrix"
    ],
    "constraints": [
      "High memory space O(n^2) and time complexity O(n^3)"
    ],
    "best_use_cases": [
      "Gene expression analysis",
      "Taxonomy tree building"
    ],
    "time_complexity": "O(n^2 log n) to O(n^3)",
    "space_complexity": "O(n^2)",
    "advantages": [
      "Produces informative dendrogram visualization",
      "No K choice required beforehand"
    ],
    "limitations": [
      "Computationally expensive for n > 10,000"
    ],
    "alternatives": [
      "K-Means Clustering"
    ],
    "python_template": "from sklearn.cluster import AgglomerativeClustering\ncluster = AgglomerativeClustering(n_clusters=3)\nlabels = cluster.fit_predict(X)"
  },
  {
    "id": "pca_dimensionality_reduction",
    "name": "Principal Component Analysis (PCA)",
    "category": "Machine Learning / AI",
    "description": "Linear dimensionality reduction technique projecting data onto principal axes of maximum variance.",
    "problem_patterns": [
      "pca dimensionality reduction",
      "feature compression maximum variance",
      "2d 3d data visualization pca"
    ],
    "data_structures": [
      "Covariance Matrix",
      "Eigenvectors / Singular Values"
    ],
    "requirements": [
      "Centered and standardized numeric features"
    ],
    "input_characteristics": [
      "High-dimensional correlated numeric features"
    ],
    "constraints": [
      "Linear transformation only"
    ],
    "best_use_cases": [
      "Reducing feature dimensions before ML training",
      "Data visualization in 2D/3D"
    ],
    "time_complexity": "O(d^2 * n + d^3)",
    "space_complexity": "O(d^2)",
    "advantages": [
      "Removes feature correlation",
      "Reduces memory and training time"
    ],
    "limitations": [
      "Principal components are less interpretable"
    ],
    "alternatives": [
      "t-SNE",
      "UMAP",
      "Truncated SVD"
    ],
    "python_template": "from sklearn.decomposition import PCA\npca = PCA(n_components=2)\nX_reduced = pca.fit_transform(X)"
  },
  {
    "id": "naive_bayes_ml",
    "name": "Naive Bayes (ML / AI)",
    "category": "Machine Learning / AI",
    "description": "Probabilistic machine learning model based on Bayes theorem applied to text and tabular classification.",
    "problem_patterns": [
      "text classification naive bayes",
      "probabilistic sentiment model",
      "document topic classification"
    ],
    "data_structures": [
      "Probability Map"
    ],
    "requirements": [
      "Preprocessed features"
    ],
    "input_characteristics": [
      "Text feature vectors (TF-IDF/Word Counts)"
    ],
    "constraints": [
      "Independence assumption"
    ],
    "best_use_cases": [
      "Spam filtering",
      "Text sentiment classification"
    ],
    "time_complexity": "O(n * d)",
    "space_complexity": "O(classes * d)",
    "advantages": [
      "Extremely fast",
      "Works well on small text training sets"
    ],
    "limitations": [
      "Independent feature assumption"
    ],
    "alternatives": [
      "Logistic Regression",
      "SVM"
    ],
    "python_template": "from sklearn.naive_bayes import MultinomialNB\nmodel = MultinomialNB()\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "mlp_neural_network",
    "name": "Multilayer Perceptron (MLP Deep Learning)",
    "category": "Machine Learning / AI",
    "description": "Feedforward artificial neural network consisting of input, hidden, and output layers with non-linear activation functions.",
    "problem_patterns": [
      "deep neural network tabular",
      "feedforward multilayer perceptron",
      "non linear neural net classification"
    ],
    "data_structures": [
      "Weight Matrices",
      "Bias Vectors"
    ],
    "requirements": [
      "Scaled inputs",
      "Backpropagation optimizer"
    ],
    "input_characteristics": [
      "Tabular or dense numeric vector features"
    ],
    "constraints": [
      "Black-box model requiring careful hyperparameter tuning"
    ],
    "best_use_cases": [
      "Complex non-linear classification/regression",
      "Function approximation"
    ],
    "time_complexity": "O(epochs * n * sum(w_i))",
    "space_complexity": "O(total_weights)",
    "advantages": [
      "Universal function approximator"
    ],
    "limitations": [
      "Prone to overfitting",
      "Requires significant training data"
    ],
    "alternatives": [
      "Random Forest Classifier",
      "XGBoost Classifier"
    ],
    "python_template": "from sklearn.neural_network import MLPClassifier\nmodel = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)\nmodel.fit(X_train, y_train)"
  },
  {
    "id": "cnn_deep_learning",
    "name": "Convolutional Neural Network (CNN)",
    "category": "Machine Learning / AI",
    "description": "Deep learning architecture utilizing convolutional filters to extract spatial feature hierarchies from visual inputs.",
    "problem_patterns": [
      "image classification cnn",
      "computer vision object detection",
      "spatial grid feature extraction"
    ],
    "data_structures": [
      "Convolutional Kernels",
      "Feature Maps"
    ],
    "requirements": [
      "GPU hardware recommended",
      "Image tensors"
    ],
    "input_characteristics": [
      "2D/3D Image grids or spatial signals"
    ],
    "constraints": [
      "High compute power required for training"
    ],
    "best_use_cases": [
      "Image classification",
      "Medical scan analysis",
      "Object recognition"
    ],
    "time_complexity": "O(convolutions * pixels * channels)",
    "space_complexity": "O(model_parameters)",
    "advantages": [
      "State-of-the-art for visual and spatial image data"
    ],
    "limitations": [
      "Computationally expensive"
    ],
    "alternatives": [
      "Vision Transformer (ViT)"
    ],
    "python_template": "# PyTorch/Keras Convolutional Network with Conv2D, MaxPooling2D, Dense layers."
  },
  {
    "id": "rnn_deep_learning",
    "name": "Recurrent Neural Network (RNN)",
    "category": "Machine Learning / AI",
    "description": "Neural network architecture with recurrent loops allowing sequential state persistence for time series and text.",
    "problem_patterns": [
      "sequential text prediction",
      "time series recurrent network",
      "sequence to sequence model"
    ],
    "data_structures": [
      "Hidden State Tensor",
      "Recurrent Weights"
    ],
    "requirements": [
      "Sequential inputs (tensors of shape [batch, seq_len, features])"
    ],
    "input_characteristics": [
      "Time series or text sequences"
    ],
    "constraints": [
      "Vanishing/exploding gradient during long sequences"
    ],
    "best_use_cases": [
      "Short sequence prediction",
      "Basic time series forecasting"
    ],
    "time_complexity": "O(seq_len * hidden_dim^2)",
    "space_complexity": "O(hidden_dim)",
    "advantages": [
      "Processes variable-length sequential inputs"
    ],
    "limitations": [
      "Vanishing gradient limits long-term memory"
    ],
    "alternatives": [
      "LSTM",
      "Transformer"
    ],
    "python_template": "# Recurrent Neural Network layer processing sequence inputs step-by-step."
  },
  {
    "id": "lstm_deep_learning",
    "name": "Long Short-Term Memory (LSTM)",
    "category": "Machine Learning / AI",
    "description": "Specialized RNN architecture featuring memory cell gates (input, forget, output) to capture long-term sequential dependencies.",
    "problem_patterns": [
      "lstm sequence prediction",
      "time series stock forecasting lstm",
      "long term sequence memory"
    ],
    "data_structures": [
      "Cell State",
      "Gate Tensors (Forget, Input, Output)"
    ],
    "requirements": [
      "Sequential tensor inputs"
    ],
    "input_characteristics": [
      "Sequential time series data or long text passages"
    ],
    "constraints": [
      "Sequential processing limits parallel GPU scaling"
    ],
    "best_use_cases": [
      "Financial time series forecasting",
      "Speech recognition",
      "Sensory telemetry stream analysis"
    ],
    "time_complexity": "O(seq_len * 4 * hidden_dim^2)",
    "space_complexity": "O(hidden_dim)",
    "advantages": [
      "Prevents vanishing gradient",
      "Remembers long-term temporal dependencies"
    ],
    "limitations": [
      "Slower training than Transformers due to sequential time step dependency"
    ],
    "alternatives": [
      "GRU",
      "Transformer"
    ],
    "python_template": "# PyTorch nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)"
  },
  {
    "id": "transformer_network",
    "name": "Transformer Network",
    "category": "Machine Learning / AI",
    "description": "Deep learning architecture based entirely on self-attention mechanisms, powering modern Large Language Models (LLMs).",
    "problem_patterns": [
      "transformer llm text processing",
      "self attention sequence model",
      "bert gpt natural language model"
    ],
    "data_structures": [
      "Multi-Head Attention Matrices",
      "Position Embeddings"
    ],
    "requirements": [
      "GPU/TPU hardware acceleration",
      "Tokenized text embeddings"
    ],
    "input_characteristics": [
      "Text tokens, sequences, multi-modal data"
    ],
    "constraints": [
      "Quadratic O(L^2) attention memory cost without sparse attention"
    ],
    "best_use_cases": [
      "Large Language Models (LLMs)",
      "Machine translation",
      "Document summarization"
    ],
    "time_complexity": "O(L^2 * d)",
    "space_complexity": "O(L^2 + L * d)",
    "advantages": [
      "Highly parallelizable training",
      "State-of-the-art NLP context understanding"
    ],
    "limitations": [
      "Requires massive data and compute for pre-training"
    ],
    "alternatives": [
      "LSTM",
      "State Space Models (Mamba)"
    ],
    "python_template": "from transformers import AutoModel, AutoTokenizer\ntokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\nmodel = AutoModel.from_pretrained('bert-base-uncased')"
  },
  {
    "id": "content_based_recommendation",
    "name": "Content-Based Recommendation",
    "category": "Machine Learning / AI",
    "description": "Recommender system matching item feature attributes (genres, keywords, descriptions) against user profile preferences.",
    "problem_patterns": [
      "content based recommender",
      "item attribute recommendation",
      "similar movie recommendation by genre"
    ],
    "data_structures": [
      "Item Feature Vectors",
      "User Profile Vectors"
    ],
    "requirements": [
      "Item metadata features (text, tags, categories)"
    ],
    "input_characteristics": [
      "Item metadata and user rating history"
    ],
    "constraints": [
      "Cannot recommend items outside user's historical feature exposure"
    ],
    "best_use_cases": [
      "Movie/article recommendations based on item metadata features"
    ],
    "time_complexity": "O(users * items * features)",
    "space_complexity": "O(items * features)",
    "advantages": [
      "No cold-start problem for new items with rich metadata"
    ],
    "limitations": [
      "Over-specialization, limited novelty"
    ],
    "alternatives": [
      "Collaborative Filtering Recommendation",
      "Hybrid Recommendation"
    ],
    "python_template": "from sklearn.metrics.pairwise import cosine_similarity\nitem_similarities = cosine_similarity(item_feature_matrix)"
  },
  {
    "id": "collaborative_filtering",
    "name": "Collaborative Filtering Recommendation",
    "category": "Machine Learning / AI",
    "description": "Recommender system technique filtering items based on user-item interaction matrix similarities (User-User or Item-Item SVD).",
    "problem_patterns": [
      "collaborative filtering movie recommender",
      "matrix factorization svd recommendation",
      "user item interaction matrix"
    ],
    "data_structures": [
      "User-Item Interaction Matrix",
      "Latent Factor Matrices (U, V)"
    ],
    "requirements": [
      "User interaction matrix (ratings, clicks, views)"
    ],
    "input_characteristics": [
      "Sparse user-item rating matrix"
    ],
    "constraints": [
      "Cold start problem for new users or new unrated items"
    ],
    "best_use_cases": [
      "E-commerce product recommendations ('Users who bought X also bought Y')"
    ],
    "time_complexity": "O(k * non_zero_ratings)",
    "space_complexity": "O((users + items) * k)",
    "advantages": [
      "Discovers serendipitous recommendations without requiring manual item metadata"
    ],
    "limitations": [
      "Cold start problem for new users/items"
    ],
    "alternatives": [
      "Content-Based Recommendation",
      "Hybrid Recommendation"
    ],
    "python_template": "from scipy.sparse.linalg import svds\nU, sigma, Vt = svds(user_item_matrix, k=20)"
  },
  {
    "id": "hybrid_recommendation",
    "name": "Hybrid Recommendation System",
    "category": "Machine Learning / AI",
    "description": "Recommender system combining collaborative filtering and content-based approaches to maximize accuracy and coverage.",
    "problem_patterns": [
      "hybrid recommender system",
      "combined content collaborative recommendation"
    ],
    "data_structures": [
      "Ensemble Scoring Engine"
    ],
    "requirements": [
      "Both user-item interactions and item metadata"
    ],
    "input_characteristics": [
      "User interaction logs and item feature metadata"
    ],
    "constraints": [
      "System design and tuning complexity"
    ],
    "best_use_cases": [
      "Production recommendation platforms (Netflix, Amazon, Spotify)"
    ],
    "time_complexity": "O(content_time + collaborative_time)",
    "space_complexity": "O(content_space + collaborative_space)",
    "advantages": [
      "Mitigates cold-start problem while preserving serendipity"
    ],
    "limitations": [
      "Higher engineering complexity"
    ],
    "alternatives": [
      "Collaborative Filtering",
      "Two-Tower Neural Recommenders"
    ],
    "python_template": "# Combines weighted scores: final_score = alpha * content_score + (1 - alpha) * collaborative_score"
  }
]
