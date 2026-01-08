# Algorithms & Data Structures – Master Cheat Sheet

> **For FAANG-Level Interview Preparation**  
> **Complete Reference | Pattern-Based Approach | Production-Ready**

---

# Table of Contents


1. Array & String Fundamentals
2. Hashing / Hash Maps / Sets
3. Two Pointers
4. Sliding Window (fixed & variable)
5. Prefix Sum / Difference Array
6. Stack
7. Queue / Deque / Monotonic Queue
8. Linked List (all variants)
9. Binary Search (classic + on answer)
10. Recursion & Backtracking
11. Tree Traversals (DFS/BFS)
12. Binary Tree Patterns
13. Binary Search Tree
14. Heap / Priority Queue
15. Greedy Algorithms
16. Graphs:
    - BFS / DFS
    - Topological Sort
    - Union-Find
    - Shortest Path (Dijkstra, Bellman-Ford)
17. Dynamic Programming:
    - 1D DP
    - 2D DP
    - Knapsack patterns
    - Subsequence / Subarray DP
    - DP optimization
18. Bit Manipulation
19. Trie
20. Segment Tree / Fenwick Tree
21. Advanced Patterns:
    - Divide & Conquer
    - Sweep Line
    - Interval Problems
    - Monotonic Stack

---

## 1. Array & String Fundamentals

---

### ❓ When should I use this?

- Problem involves iterating through elements
- Need to track indices, counts, or frequencies
- Keywords: "subarray", "contiguous", "in-place", "rotate", "reverse"
- When brute force involves nested loops but optimization is needed

### 🧠 Core Idea (Intuition)

Arrays are contiguous memory blocks with O(1) access. Most optimizations involve:
- **Avoiding nested loops** through clever iteration
- **Using extra space** (hash maps) to trade time for space
- **Two-pass algorithms** (forward + backward)
- **In-place modifications** to achieve O(1) space

### 🧩 Common Problem Types

- Find duplicates/missing elements
- Rotate or reverse arrays
- Merge sorted arrays
- Product/sum calculations excluding current element
- Rearrange elements based on conditions

### 🧱 Template (Python)

```python
# Pattern 1: Single Pass with Hash Map
def solve_with_hashmap(arr):
    seen = {}  # or set() if only existence matters
    result = []
    
    for i, num in enumerate(arr):
        # Check condition using seen
        if some_condition(num, seen):
            result.append(num)
        
        # Update seen
        seen[num] = i  # or seen.add(num)
    
    return result

# Pattern 2: Two Pass (Forward + Backward)
def solve_two_pass(arr):
    n = len(arr)
    left = [0] * n
    right = [0] * n
    
    # Forward pass
    for i in range(n):
        left[i] = compute_left(arr, i, left)
    
    # Backward pass
    for i in range(n - 1, -1, -1):
        right[i] = compute_right(arr, i, right)
    
    # Combine results
    return [combine(left[i], right[i]) for i in range(n)]

# Pattern 3: In-Place Modification
def solve_inplace(arr):
    # Use array indices as hash keys
    for i in range(len(arr)):
        index = abs(arr[i]) - 1  # Map value to index
        arr[index] = -abs(arr[index])  # Mark as visited
    
    # Extract result from modified array
    result = [i + 1 for i in range(len(arr)) if arr[i] > 0]
    return result
```

### 📌 Step-by-Step Walkthrough

**Example: Find all duplicates in array [4,3,2,7,8,2,3,1]**

1. Use indices as markers (in-place hashing)
2. For each number `num`, go to index `num-1`
3. Mark that position negative
4. If already negative → duplicate found

```
Initial: [4,3,2,7,8,2,3,1]
i=0, num=4: [4,3,2,-7,8,2,3,1]
i=1, num=3: [4,3,-2,-7,8,2,3,1]
i=2, num=2: [4,-3,-2,-7,8,2,3,1]
i=3, num=7: [4,-3,-2,-7,8,2,-3,1]
i=4, num=8: [4,-3,-2,-7,8,2,-3,-1]
i=5, num=2: already negative at index 1 → duplicate: 2
i=6, num=3: already negative at index 2 → duplicate: 3
```

### 🧪 Solved Examples

**1. Two Sum (Hash Map)**
```python
def twoSum(nums, target):
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
```

**2. Product of Array Except Self**
```python
def productExceptSelf(nums):
    n = len(nums)
    result = [1] * n
    
    # Left products
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]
    
    # Right products
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]
    
    return result
```

**3. Find All Duplicates (In-Place)**
```python
def findDuplicates(nums):
    duplicates = []
    for num in nums:
        index = abs(num) - 1
        if nums[index] < 0:
            duplicates.append(abs(num))
        else:
            nums[index] = -nums[index]
    return duplicates
```

**4. Rotate Array**
```python
def rotate(nums, k):
    k = k % len(nums)
    # Reverse entire array
    reverse(nums, 0, len(nums) - 1)
    # Reverse first k elements
    reverse(nums, 0, k - 1)
    # Reverse remaining elements
    reverse(nums, k, len(nums) - 1)

def reverse(nums, start, end):
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1
```

### ⚠️ Edge Cases & Pitfalls

- **Empty array**: Always check `if not arr` or `if len(arr) == 0`
- **Single element**: Many algorithms fail with n=1
- **Duplicates**: Does problem allow duplicates? Changes approach
- **Negative numbers**: In-place marking breaks with negatives
- **Integer overflow**: When multiplying/adding large numbers
- **Index out of bounds**: Off-by-one errors in loops
- **Modifying while iterating**: Use `arr[:]` or iterate backwards

### ⏱️ Time & Space Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Access | O(1) | - |
| Search (unsorted) | O(n) | - |
| Search (sorted) | O(log n) | - |
| Insert/Delete (end) | O(1) | - |
| Insert/Delete (middle) | O(n) | - |
| Hash Map approach | O(n) | O(n) |
| Two-pass algorithm | O(n) | O(1) or O(n) |

### 🧠 Interview Tips

- **Always clarify**: sorted? duplicates? negative numbers?
- **Start with brute force**: O(n²) → explain → optimize to O(n)
- **Think out loud**: "I could use a hash map to store..."
- **Space-time tradeoff**: Mention O(n) space solution first, then O(1) if possible
- **Test with example**: Walk through small array [1,2,3]
- **Edge cases to mention**: empty, single element, all same, all different

---

## 2. Hashing / Hash Maps / Sets

---

### ❓ When should I use this?

- Need **O(1) lookup** of previously seen elements
- Keywords: "find pair", "count frequency", "check existence", "group by"
- When you need to **avoid nested loops** (replace inner loop with hash lookup)
- Tracking state: visited nodes, seen characters, counts

### 🧠 Core Idea (Intuition)

Hash tables trade **space for time**:
- Instead of searching through array (O(n)), store in hash map for O(1) lookup
- **Key insight**: If you're checking "have I seen X before?", use a set
- If you need to store **associated data** with X, use a dictionary/map

**Mental model**: Think of it as a magical instant-lookup notebook where you can write and read in constant time.

### 🧩 Common Problem Types

- Two Sum / K Sum problems
- Frequency counting (anagrams, character counts)
- Grouping elements (group anagrams, group by property)
- Detecting cycles (linked list, array indices)
- Subarray sum problems (with prefix sums)
- First unique character / first non-repeating

### 🧱 Template (Python)

```python
# Pattern 1: Existence Check (Set)
def pattern_existence(arr):
    seen = set()
    for item in arr:
        if item in seen:
            return True  # Found duplicate/pair
        seen.add(item)
    return False

# Pattern 2: Counting Frequency (Counter)
from collections import Counter

def pattern_frequency(arr):
    freq = Counter(arr)  # or freq = {}
    # Alternative manual counting:
    # for item in arr:
    #     freq[item] = freq.get(item, 0) + 1
    
    for item, count in freq.items():
        if count > threshold:
            return item

# Pattern 3: Grouping (Defaultdict)
from collections import defaultdict

def pattern_grouping(items):
    groups = defaultdict(list)
    
    for item in items:
        key = compute_key(item)  # e.g., sorted string for anagrams
        groups[key].append(item)
    
    return list(groups.values())

# Pattern 4: Index Mapping (Value -> Index)
def pattern_index_map(arr, target):
    index_map = {}  # value -> index
    
    for i, val in enumerate(arr):
        complement = target - val
        if complement in index_map:
            return [index_map[complement], i]
        index_map[val] = i

# Pattern 5: Sliding Window with Hash Map
def pattern_window_hash(s, k):
    window = {}
    result = []
    
    for i, char in enumerate(s):
        # Add to window
        window[char] = window.get(char, 0) + 1
        
        # Shrink window if needed
        if i >= k:
            left_char = s[i - k]
            window[left_char] -= 1
            if window[left_char] == 0:
                del window[left_char]
        
        # Check window condition
        if len(window) == k:
            result.append(i)
    
    return result
```

### 📌 Step-by-Step Walkthrough

**Example: Two Sum with [2,7,11,15], target=9**

```
Initialize: seen = {}, target = 9

i=0, num=2:
  complement = 9 - 2 = 7
  7 not in seen
  seen = {2: 0}

i=1, num=7:
  complement = 9 - 7 = 2
  2 in seen! ✓
  return [seen[2], 1] = [0, 1]
```

**Example: Group Anagrams ["eat","tea","tan","ate","nat","bat"]**

```
groups = {}

"eat" → sorted = "aet" → groups = {"aet": ["eat"]}
"tea" → sorted = "aet" → groups = {"aet": ["eat","tea"]}
"tan" → sorted = "ant" → groups = {"aet": [...], "ant": ["tan"]}
"ate" → sorted = "aet" → groups = {"aet": ["eat","tea","ate"], "ant": [...]}
"nat" → sorted = "ant" → groups = {"aet": [...], "ant": ["tan","nat"]}
"bat" → sorted = "abt" → groups = {"aet": [...], "ant": [...], "abt": ["bat"]}

Result: [["eat","tea","ate"], ["tan","nat"], ["bat"]]
```

### 🧪 Solved Examples

**1. Two Sum**
```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

**2. Group Anagrams**
```python
def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))  # or use tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```

**3. First Unique Character**
```python
def firstUniqChar(s):
    freq = Counter(s)
    for i, char in enumerate(s):
        if freq[char] == 1:
            return i
    return -1
```

**4. Subarray Sum Equals K**
```python
def subarraySum(nums, k):
    count = 0
    prefix_sum = 0
    sum_freq = {0: 1}  # prefix_sum -> frequency
    
    for num in nums:
        prefix_sum += num
        # Check if (prefix_sum - k) exists
        if prefix_sum - k in sum_freq:
            count += sum_freq[prefix_sum - k]
        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1
    
    return count
```

**5. Longest Substring Without Repeating Characters**
```python
def lengthOfLongestSubstring(s):
    char_index = {}
    max_len = 0
    start = 0
    
    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = end
        max_len = max(max_len, end - start + 1)
    
    return max_len
```

**6. Isomorphic Strings**
```python
def isIsomorphic(s, t):
    map_s_to_t = {}
    map_t_to_s = {}
    
    for char_s, char_t in zip(s, t):
        if char_s in map_s_to_t:
            if map_s_to_t[char_s] != char_t:
                return False
        else:
            map_s_to_t[char_s] = char_t
        
        if char_t in map_t_to_s:
            if map_t_to_s[char_t] != char_s:
                return False
        else:
            map_t_to_s[char_t] = char_s
    
    return True
```

### ⚠️ Edge Cases & Pitfalls

- **Hash collision**: Python handles this, but be aware in other languages
- **Mutable keys**: Lists/sets can't be dictionary keys (use tuples)
- **Default values**: Use `dict.get(key, default)` or `defaultdict`
- **Deleting while iterating**: Create list of keys first
- **Checking before access**: Always check `if key in dict` before accessing
- **Counter edge case**: `Counter` returns 0 for missing keys (no KeyError)
- **Memory**: Hash maps use O(n) space - mention this in interviews
- **Ordering**: Regular dicts maintain insertion order (Python 3.7+)

### ⏱️ Time & Space Complexity

| Operation | Average | Worst Case | Notes |
|-----------|---------|------------|-------|
| Insert | O(1) | O(n) | Worst case: hash collisions |
| Lookup | O(1) | O(n) | Amortized O(1) |
| Delete | O(1) | O(n) | Same as lookup |
| Space | O(n) | O(n) | n = number of elements |

**Common complexities:**
- Two Sum: O(n) time, O(n) space
- Group Anagrams: O(n * k log k) time where k = max string length
- Frequency counting: O(n) time, O(n) space

### 🧠 Interview Tips

- **Always clarify input constraints**: size, character set, duplicates allowed?
- **Mention space tradeoff**: "I'm using O(n) extra space for the hash map"
- **Explain hash function**: For custom keys (anagrams), explain why you chose that key
- **Walk through example**: Show how hash map builds up step by step
- **Optimization**: If interviewer asks to reduce space, discuss if possible
- **Common follow-up**: "What if we can't use extra space?" → Different approach needed
- **Python-specific**: Mention Counter, defaultdict, OrderedDict when relevant
- **Collision handling**: If asked, explain chaining vs open addressing

**Red flags to avoid:**
- Not checking if key exists before accessing
- Using mutable objects as keys
- Forgetting to handle empty input
- Not considering hash map overhead for tiny inputs

---

## 3. Two Pointers

---

### ❓ When should I use this?

- Array/string is **sorted** or can be sorted
- Need to find **pairs** or **triplets** with certain properties
- Keywords: "pair", "triplet", "opposite ends", "partition", "remove duplicates"
- When brute force requires O(n²) nested loops but can be optimized
- **Opposite direction**: Start from both ends moving inward
- **Same direction**: Both pointers move left-to-right (slow/fast)

### 🧠 Core Idea (Intuition)

**Two pointers eliminates one nested loop** by using problem structure:

1. **Opposite direction** (left/right): Sorted array lets you make decisions
   - If sum too large → move right pointer left
   - If sum too small → move left pointer right

2. **Same direction** (slow/fast): Process array in one pass
   - Slow pointer: where to write/valid position
   - Fast pointer: exploring ahead

**Mental model**: Like two people searching from opposite ends of a bookshelf or one person moving items while another scans ahead.

### 🧩 Common Problem Types

- Two Sum (sorted array)
- Three Sum / Four Sum
- Container with most water
- Trapping rain water
- Remove duplicates from sorted array
- Partition array (Dutch National Flag)
- Palindrome checking
- Merge sorted arrays
- Linked list cycle detection (fast/slow)

### 🧱 Template (Python)

```python
# Pattern 1: Opposite Direction (Left-Right)
def two_pointers_opposite(arr):
    left, right = 0, len(arr) - 1
    result = []
    
    while left < right:
        # Calculate current state
        current = compute(arr[left], arr[right])
        
        if condition_met(current):
            result.append([arr[left], arr[right]])
            left += 1
            right -= 1
        elif current < target:
            left += 1  # Need larger value
        else:
            right -= 1  # Need smaller value
    
    return result

# Pattern 2: Same Direction (Slow-Fast)
def two_pointers_same_direction(arr):
    slow = 0  # Write position / valid boundary
    
    for fast in range(len(arr)):
        if is_valid(arr[fast]):
            arr[slow] = arr[fast]
            slow += 1
    
    return slow  # or arr[:slow]

# Pattern 3: Sliding Window with Two Pointers
def two_pointers_window(arr, target):
    left = 0
    current_sum = 0
    result = []
    
    for right in range(len(arr)):
        current_sum += arr[right]
        
        # Shrink window from left
        while current_sum > target and left <= right:
            current_sum -= arr[left]
            left += 1
        
        if current_sum == target:
            result.append([left, right])
    
    return result

# Pattern 4: Partition (Three Pointers - Dutch National Flag)
def three_way_partition(arr, pivot):
    low, mid, high = 0, 0, len(arr) - 1
    
    while mid <= high:
        if arr[mid] < pivot:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == pivot:
            mid += 1
        else:  # arr[mid] > pivot
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
    
    return arr
```

### 📌 Step-by-Step Walkthrough

**Example 1: Two Sum (sorted) [2,7,11,15], target=9**

```
Initial: left=0, right=3
arr[0]=2, arr[3]=15, sum=17 > 9 → right--

Step 2: left=0, right=2
arr[0]=2, arr[2]=11, sum=13 > 9 → right--

Step 3: left=0, right=1
arr[0]=2, arr[1]=7, sum=9 ✓ → found!
```

**Example 2: Remove Duplicates [1,1,2,2,3]**

```
slow=0, fast=0: arr[0]=1, write arr[0]=1, slow=1
slow=1, fast=1: arr[1]=1 (duplicate), skip
slow=1, fast=2: arr[2]=2, write arr[1]=2, slow=2
slow=2, fast=3: arr[3]=2 (duplicate), skip
slow=2, fast=4: arr[4]=3, write arr[2]=3, slow=3

Result: [1,2,3,_,_], return slow=3
```

**Example 3: Dutch National Flag [2,0,2,1,1,0], pivot=1**

```
low=0, mid=0, high=5

mid=0: arr[0]=2 > 1 → swap with high → [0,0,2,1,1,2], high=4
mid=0: arr[0]=0 < 1 → swap with low → [0,0,2,1,1,2], low=1, mid=1
mid=1: arr[1]=0 < 1 → swap with low → [0,0,2,1,1,2], low=2, mid=2
mid=2: arr[2]=2 > 1 → swap with high → [0,0,1,1,2,2], high=3
mid=2: arr[2]=1 = 1 → mid=3
mid=3: arr[3]=1 = 1 → mid=4
mid=4: mid > high, stop

Result: [0,0,1,1,2,2]
```

### 🧪 Solved Examples

**1. Two Sum II (Sorted)**
```python
def twoSum(numbers, target):
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []
```

**2. Three Sum**
```python
def threeSum(nums):
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        target = -nums[i]
        
        while left < right:
            current_sum = nums[left] + nums[right]
            
            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates for second number
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                # Skip duplicates for third number
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return result
```

**3. Container With Most Water**
```python
def maxArea(height):
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        current_area = width * min(height[left], height[right])
        max_area = max(max_area, current_area)
        
        # Move pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area
```

**4. Remove Duplicates from Sorted Array**
```python
def removeDuplicates(nums):
    if not nums:
        return 0
    
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1
```

**5. Sort Colors (Dutch National Flag)**
```python
def sortColors(nums):
    low, mid, high = 0, 0, len(nums) - 1
    
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
```

**6. Valid Palindrome**
```python
def isPalindrome(s):
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True
```

**7. Trapping Rain Water**
```python
def trap(height):
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    water = 0
    
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            water += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            water += right_max - height[right]
    
    return water
```

### ⚠️ Edge Cases & Pitfalls

- **Empty array**: Check `if not arr` or `len(arr) == 0`
- **Single element**: Many two-pointer algorithms need at least 2 elements
- **Duplicate handling**: Three Sum requires careful duplicate skipping
- **Pointer crossing**: Ensure `left < right`, not `left <= right` (usually)
- **Off-by-one**: When returning indices, check if 0-indexed or 1-indexed
- **Infinite loops**: Make sure pointers always progress
- **Sorted requirement**: Some problems require sorting first (O(n log n))
- **In-place modification**: Clarify if you can modify input array

### ⏱️ Time & Space Complexity

| Pattern | Time | Space | Notes |
|---------|------|-------|-------|
| Two Sum (sorted) | O(n) | O(1) | Single pass |
| Three Sum | O(n²) | O(1) | O(n log n) for sort + O(n²) for pairs |
| Remove duplicates | O(n) | O(1) | In-place |
| Container with water | O(n) | O(1) | Single pass |
| Dutch National Flag | O(n) | O(1) | Single pass, 3 partitions |
| Trapping rain water | O(n) | O(1) | Two-pointer approach |

**General principle**: Two pointers reduces O(n²) to O(n) by eliminating one loop.

### 🧠 Interview Tips

- **Clarify if sorted**: "Can I assume the array is sorted?" If not, "Can I sort it?"
- **Draw it out**: Sketch array with two arrows showing pointer movement
- **Explain decision logic**: "I move left pointer when sum is too small because..."
- **Mention optimization**: "Brute force would be O(n²), but two pointers gives O(n)"
- **Handle duplicates carefully**: In Three Sum, explain why you skip duplicates
- **Edge case walkthrough**: Show what happens with [1,2] or empty array
- **Space complexity**: Emphasize O(1) space (excluding output)

**Common follow-ups:**
- "What if array is not sorted?" → Need to sort first or use different approach
- "Can you do it in-place?" → Yes, two pointers excel at in-place operations
- "What about Four Sum?" → Add another loop, becomes O(n³)

**Red flags to avoid:**
- Forgetting to move pointers (infinite loop)
- Not handling duplicates in Three Sum
- Confusing when to move which pointer
- Off-by-one errors in indices

---

## 4. Sliding Window (Fixed & Variable)

---

### ❓ When should I use this?

- Problem involves **contiguous subarrays/substrings**
- Keywords: "substring", "subarray", "consecutive", "window", "k elements"
- Need to find **maximum/minimum/count** over all windows
- Optimization: Avoid recalculating from scratch for each window

**Two types:**
1. **Fixed size**: Window size k is given
2. **Variable size**: Window size changes based on condition

### 🧠 Core Idea (Intuition)

**Fixed window**: Slide a window of size k, update result as you go
- Add new element on right
- Remove old element on left
- Update answer

**Variable window**: Expand/shrink window to maintain condition
- Expand: Add elements while condition valid
- Shrink: Remove elements when condition violated
- Track best result

**Mental model**: Like looking through a moving frame on a film strip. Instead of analyzing each frame from scratch, you update based on what enters/exits the frame.

### 🧩 Common Problem Types

**Fixed size:**
- Maximum sum of subarray of size k
- Average of subarrays of size k
- Maximum of all subarrays of size k

**Variable size:**
- Longest substring without repeating characters
- Longest substring with at most k distinct characters
- Minimum window substring
- Longest subarray with sum ≤ k
- Fruits into baskets

### 🧱 Template (Python)

```python
# Pattern 1: Fixed Size Window
def fixed_window(arr, k):
    if len(arr) < k:
        return None
    
    # Initialize first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide window
    for i in range(k, len(arr)):
        # Add new element, remove leftmost
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Pattern 2: Variable Size Window (Maximum)
def variable_window_max(arr, condition):
    left = 0
    max_length = 0
    window_state = {}  # Track window state
    
    for right in range(len(arr)):
        # Expand window: add arr[right]
        update_window_add(window_state, arr[right])
        
        # Shrink window while condition violated
        while not is_valid(window_state, condition):
            update_window_remove(window_state, arr[left])
            left += 1
        
        # Update result
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Pattern 3: Variable Size Window (Minimum)
def variable_window_min(arr, condition):
    left = 0
    min_length = float('inf')
    window_state = {}
    
    for right in range(len(arr)):
        # Expand window
        update_window_add(window_state, arr[right])
        
        # Shrink window while condition met
        while is_valid(window_state, condition):
            min_length = min(min_length, right - left + 1)
            update_window_remove(window_state, arr[left])
            left += 1
    
    return min_length if min_length != float('inf') else 0

# Pattern 4: Variable Window with Counter
from collections import Counter

def variable_window_counter(s, target):
    left = 0
    window = Counter()
    required = Counter(target)
    formed = 0  # Number of unique chars with desired frequency
    required_chars = len(required)
    result = []
    
    for right in range(len(s)):
        char = s[right]
        window[char] += 1
        
        # Check if frequency matches
        if char in required and window[char] == required[char]:
            formed += 1
        
        # Shrink window
        while formed == required_chars and left <= right:
            # Record result
            result.append([left, right])
            
            # Remove from left
            char = s[left]
            window[char] -= 1
            if char in required and window[char] < required[char]:
                formed -= 1
            left += 1
    
    return result
```

### 📌 Step-by-Step Walkthrough

**Example 1: Maximum Sum Subarray of Size K=3 in [2,1,5,1,3,2]**

```
Initial window [2,1,5]: sum = 8

Slide right:
Remove 2, Add 1: [1,5,1], sum = 8 - 2 + 1 = 7
Remove 1, Add 3: [5,1,3], sum = 7 - 1 + 3 = 9 ← max
Remove 5, Add 2: [1,3,2], sum = 9 - 5 + 2 = 6

Maximum = 9
```

**Example 2: Longest Substring Without Repeating Characters "abcabcbb"**

```
left=0, right=0: "a", window={a:1}, len=1
left=0, right=1: "ab", window={a:1,b:1}, len=2
left=0, right=2: "abc", window={a:1,b:1,c:1}, len=3
left=0, right=3: "abca", duplicate 'a'!
  → Shrink: remove 'a', left=1
  → "bca", window={b:1,c:1,a:1}, len=3
left=1, right=4: "bcab", duplicate 'b'!
  → Shrink: remove 'b', left=2
  → "cab", window={c:1,a:1,b:1}, len=3
...continue...

Maximum length = 3
```

### 🧪 Solved Examples

**1. Maximum Sum Subarray of Size K**
```python
def maxSumSubarray(arr, k):
    if len(arr) < k:
        return 0
    
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

**2. Longest Substring Without Repeating Characters**
```python
def lengthOfLongestSubstring(s):
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Shrink window until no duplicates
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Alternative with hash map (stores indices)
def lengthOfLongestSubstring2(s):
    char_index = {}
    left = 0
    max_length = 0
    
    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        
        char_index[char] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

**3. Longest Substring with At Most K Distinct Characters**
```python
def lengthOfLongestSubstringKDistinct(s, k):
    if k == 0:
        return 0
    
    char_count = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Add character to window
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Shrink window if more than k distinct
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

**4. Minimum Window Substring**
```python
def minWindow(s, t):
    if not s or not t:
        return ""
    
    required = Counter(t)
    window = {}
    
    formed = 0  # Number of unique chars in window with desired frequency
    required_chars = len(required)
    
    left = 0
    min_len = float('inf')
    min_left = 0
    
    for right in range(len(s)):
        char = s[right]
        window[char] = window.get(char, 0) + 1
        
        if char in required and window[char] == required[char]:
            formed += 1
        
        # Shrink window while valid
        while formed == required_chars and left <= right:
            # Update result
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left
            
            # Remove from left
            char = s[left]
            window[char] -= 1
            if char in required and window[char] < required[char]:
                formed -= 1
            left += 1
    
    return s[min_left:min_left + min_len] if min_len != float('inf') else ""
```

**5. Maximum of All Subarrays of Size K (Sliding Window Maximum)**
```python
from collections import deque

def maxSlidingWindow(nums, k):
    if not nums:
        return []
    
    dq = deque()  # Stores indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements (they'll never be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result once window is full
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

**6. Longest Repeating Character Replacement**
```python
def characterReplacement(s, k):
    char_count = {}
    left = 0
    max_length = 0
    max_freq = 0  # Frequency of most common char in window
    
    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        max_freq = max(max_freq, char_count[s[right]])
        
        # If window_size - max_freq > k, shrink window
        window_size = right - left + 1
        if window_size - max_freq > k:
            char_count[s[left]] -= 1
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

**7. Permutation in String**
```python
def checkInclusion(s1, s2):
    if len(s1) > len(s2):
        return False
    
    s1_count = Counter(s1)
    window_count = Counter(s2[:len(s1)])
    
    if window_count == s1_count:
        return True
    
    for i in range(len(s1), len(s2)):
        # Add new character
        window_count[s2[i]] += 1
        
        # Remove old character
        left_char = s2[i - len(s1)]
        window_count[left_char] -= 1
        if window_count[left_char] == 0:
            del window_count[left_char]
        
        if window_count == s1_count:
            return True
    
    return False
```

### ⚠️ Edge Cases & Pitfalls

- **Empty input**: Check `if not arr` or `len(arr) == 0`
- **k > len(arr)**: In fixed window, handle this case
- **k = 0**: Variable window problems, what does this mean?
- **All same elements**: May affect distinct character problems
- **Window initialization**: Fixed window needs special handling for first k elements
- **Shrinking condition**: Be clear on `while condition` vs `if condition`
- **Integer overflow**: When summing large numbers
- **Negative numbers**: Sum-based problems may behave differently
- **Updating max_freq**: In character replacement, max_freq can be kept as running max

### ⏱️ Time & Space Complexity

| Problem Type | Time | Space | Notes |
|-------------|------|-------|-------|
| Fixed window | O(n) | O(1) | Single pass |
| Variable window (max) | O(n) | O(k) | k = distinct elements in window |
| Variable window (min) | O(n) | O(k) | Each element added/removed once |
| Sliding window maximum | O(n) | O(k) | Deque approach |
| Minimum window substring | O(S + T) | O(S + T) | S, T = string lengths |

**Key insight**: Although there's a nested while loop in variable window, each element is added and removed at most once → amortized O(n).

### 🧠 Interview Tips

- **Identify window type**: "Is the window size fixed or variable?"
- **State what you're tracking**: "I'll use a hash map to count character frequencies"
- **Explain shrinking logic**: "I shrink when the window violates the condition"
- **Walk through example**: Show window sliding step-by-step
- **Mention optimization**: "Instead of recalculating, I update incrementally"
- **Edge cases**: Empty string, k=0, k > length
- **Space complexity**: Clarify that hash map space is O(distinct characters), often O(26) = O(1) for lowercase letters

**Common follow-ups:**
- "Can you optimize space?" → Fixed window may not need extra space
- "What if input is stream?" → Sliding window works well for streaming data
- "Handle negative numbers?" → May need different approach for sum problems

**Red flags to avoid:**
- Recalculating window from scratch (defeats purpose)
- Incorrect shrinking condition (infinite loop or wrong answer)
- Not handling window initialization properly
- Forgetting to update result before shrinking (minimum window problems)

---

## 5. Prefix Sum / Difference Array

---

### ❓ When should I use this?

- Need to calculate **sum of subarray** multiple times
- Keywords: "range sum", "subarray sum", "cumulative", "difference"
- **Range queries**: Answer many queries about subarrays efficiently
- **Range updates**: Update ranges efficiently (difference array)
- Trade: O(n) preprocessing for O(1) query time

### 🧠 Core Idea (Intuition)

**Prefix Sum:**
- `prefix[i]` = sum of elements from index 0 to i
- **Subarray sum [L, R]** = `prefix[R] - prefix[L-1]`
- **Why it works**: Total up to R minus total up to L-1 = sum in between

**Difference Array:**
- Used for **range updates**: add value to range [L, R]
- Store differences between consecutive elements
- Update range in O(1), reconstruct array in O(n)

**Mental model**: 
- Prefix sum: Like a running total on a receipt
- Difference array: Like recording only changes in elevation on a hiking trail

### 🧩 Common Problem Types

**Prefix Sum:**
- Range sum queries
- Subarray sum equals k
- Continuous subarray sum
- Product of array except self (prefix products)
- 2D matrix range sum

**Difference Array:**
- Range addition queries
- Corporate flight bookings
- Car pooling
- Meeting rooms

### 🧱 Template (Python)

```python
# Pattern 1: Basic Prefix Sum
def build_prefix_sum(arr):
    n = len(arr)
    prefix = [0] * (n + 1)  # prefix[0] = 0 for convenience
    
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]
    
    return prefix

def range_sum(prefix, left, right):
    """Sum of arr[left:right+1] using prefix sum"""
    return prefix[right + 1] - prefix[left]

# Pattern 2: Prefix Sum with Hash Map (Subarray Sum = K)
def subarray_sum_equals_k(arr, k):
    prefix_sum = 0
    sum_count = {0: 1}  # prefix_sum -> count
    result = 0
    
    for num in arr:
        prefix_sum += num
        
        # Check if (prefix_sum - k) exists
        if prefix_sum - k in sum_count:
            result += sum_count[prefix_sum - k]
        
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
    
    return result

# Pattern 3: Difference Array (Range Updates)
def build_difference_array(arr):
    n = len(arr)
    diff = [0] * n
    
    diff[0] = arr[0]
    for i in range(1, n):
        diff[i] = arr[i] - arr[i - 1]
    
    return diff

def range_update(diff, left, right, val):
    """Add val to range [left, right]"""
    diff[left] += val
    if right + 1 < len(diff):
        diff[right + 1] -= val

def reconstruct_from_diff(diff):
    """Convert difference array back to original array"""
    n = len(diff)
    arr = [0] * n
    arr[0] = diff[0]
    
    for i in range(1, n):
        arr[i] = arr[i - 1] + diff[i]
    
    return arr

# Pattern 4: 2D Prefix Sum
def build_2d_prefix_sum(matrix):
    if not matrix or not matrix[0]:
        return []
    
    rows, cols = len(matrix), len(matrix[0])
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            prefix[i][j] = (matrix[i-1][j-1] + 
                           prefix[i-1][j] + 
                           prefix[i][j-1] - 
                           prefix[i-1][j-1])
    
    return prefix

def range_sum_2d(prefix, row1, col1, row2, col2):
    """Sum of submatrix from (row1,col1) to (row2,col2)"""
    return (prefix[row2+1][col2+1] - 
            prefix[row1][col2+1] - 
            prefix[row2+1][col1] + 
            prefix[row1][col1])

# Pattern 5: Prefix Product
def product_except_self(nums):
    n = len(nums)
    result = [1] * n
    
    # Left products
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]
    
    # Right products
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]
    
    return result
```

### 📌 Step-by-Step Walkthrough

**Example 1: Prefix Sum for [1,2,3,4,5]**

```
Array:  [1, 2, 3, 4, 5]
Prefix: [0, 1, 3, 6, 10, 15]
         ↑  ↑  ↑  ↑  ↑   ↑
         0  1  2  3  4   5 (indices)

Query: Sum of arr[1:4] (elements 2,3,4)
= prefix[4] - prefix[1]
= 10 - 1 = 9 ✓
```

**Example 2: Subarray Sum Equals K=7 in [1,2,3,4]**

```
i=0, num=1: prefix_sum=1
  Check: 1-7=-6 not in map
  sum_count = {0:1, 1:1}

i=1, num=2: prefix_sum=3
  Check: 3-7=-4 not in map
  sum_count = {0:1, 1:1, 3:1}

i=2, num=3: prefix_sum=6
  Check: 6-7=-1 not in map
  sum_count = {0:1, 1:1, 3:1, 6:1}

i=3, num=4: prefix_sum=10
  Check: 10-7=3 in map! count=1
  result = 1 (subarray [3,4])
  sum_count = {0:1, 1:1, 3:1, 6:1, 10:1}

Answer: 1 subarray
```

**Example 3: Difference Array - Add 2 to range [1,3] in [1,2,3,4,5]**

```
Original: [1, 2, 3, 4, 5]
Diff:     [1, 1, 1, 1, 1]

Update [1,3] by 2:
  diff[1] += 2 → [1, 3, 1, 1, 1]
  diff[4] -= 2 → [1, 3, 1, 1, -1]

Reconstruct:
  arr[0] = 1
  arr[1] = 1 + 3 = 4
  arr[2] = 4 + 1 = 5
  arr[3] = 5 + 1 = 6
  arr[4] = 6 + (-1) = 5

Result: [1, 4, 5, 6, 5] ✓
```

### 🧪 Solved Examples

**1. Range Sum Query (Immutable)**
```python
class NumArray:
    def __init__(self, nums):
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)
    
    def sumRange(self, left, right):
        return self.prefix[right + 1] - self.prefix[left]
```

**2. Subarray Sum Equals K**
```python
def subarraySum(nums, k):
    count = 0
    prefix_sum = 0
    sum_freq = {0: 1}
    
    for num in nums:
        prefix_sum += num
        if prefix_sum - k in sum_freq:
            count += sum_freq[prefix_sum - k]
        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1
    
    return count
```

**3. Continuous Subarray Sum (Multiple of K)**
```python
def checkSubarraySum(nums, k):
    # Use modulo as key: if same remainder appears twice,
    # the subarray between them is divisible by k
    remainder_index = {0: -1}  # remainder -> first index
    prefix_sum = 0
    
    for i, num in enumerate(nums):
        prefix_sum += num
        remainder = prefix_sum % k
        
        if remainder in remainder_index:
            if i - remainder_index[remainder] >= 2:
                return True
        else:
            remainder_index[remainder] = i
    
    return False
```

**4. Product of Array Except Self**
```python
def productExceptSelf(nums):
    n = len(nums)
    result = [1] * n
    
    # Left products
    for i in range(1, n):
        result[i] = result[i - 1] * nums[i - 1]
    
    # Right products
    right = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right
        right *= nums[i]
    
    return result
```

**5. Corporate Flight Bookings (Difference Array)**
```python
def corpFlightBookings(bookings, n):
    diff = [0] * (n + 1)
    
    for first, last, seats in bookings:
        diff[first - 1] += seats
        diff[last] -= seats
    
    # Reconstruct
    result = []
    current = 0
    for i in range(n):
        current += diff[i]
        result.append(current)
    
    return result
```

**6. Range Sum Query 2D (Immutable)**
```python
class NumMatrix:
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            return
        
        rows, cols = len(matrix), len(matrix[0])
        self.prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
        
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                self.prefix[i][j] = (matrix[i-1][j-1] + 
                                    self.prefix[i-1][j] + 
                                    self.prefix[i][j-1] - 
                                    self.prefix[i-1][j-1])
    
    def sumRegion(self, row1, col1, row2, col2):
        return (self.prefix[row2+1][col2+1] - 
                self.prefix[row1][col2+1] - 
                self.prefix[row2+1][col1] + 
                self.prefix[row1][col1])
```

**7. Contiguous Array (Equal 0s and 1s)**
```python
def findMaxLength(nums):
    # Treat 0 as -1, find longest subarray with sum 0
    sum_index = {0: -1}  # prefix_sum -> first index
    prefix_sum = 0
    max_len = 0
    
    for i, num in enumerate(nums):
        prefix_sum += 1 if num == 1 else -1
        
        if prefix_sum in sum_index:
            max_len = max(max_len, i - sum_index[prefix_sum])
        else:
            sum_index[prefix_sum] = i
    
    return max_len
```

### ⚠️ Edge Cases & Pitfalls

- **Off-by-one errors**: Prefix array is size n+1, be careful with indices
- **Empty array**: Handle `if not arr` cases
- **Single element**: Prefix sum works, but check range queries
- **Negative numbers**: Prefix sum works fine with negatives
- **Integer overflow**: Large sums may overflow (use long in Java)
- **Modulo arithmetic**: In continuous subarray sum, handle k=0 case
- **2D matrix**: Inclusion-exclusion principle (add/subtract overlaps)
- **Difference array reconstruction**: Don't forget to apply prefix sum after updates

### ⏱️ Time & Space Complexity

| Operation | Preprocessing | Query/Update | Space |
|-----------|--------------|--------------|-------|
| Prefix sum (build) | O(n) | O(1) per query | O(n) |
| Subarray sum = k | O(n) | - | O(n) |
| Difference array | O(n) | O(1) per update | O(n) |
| 2D prefix sum (build) | O(m×n) | O(1) per query | O(m×n) |
| Range updates + query | O(n + q) | O(1) per update | O(n) |

**Key tradeoff**: Spend O(n) once to enable O(1) queries

### 🧠 Interview Tips

- **Clarify query count**: "How many queries will there be?" (justifies preprocessing)
- **Explain tradeoff**: "I'll spend O(n) to build prefix sum, then each query is O(1)"
- **Draw diagram**: Show prefix array and how subtraction works
- **Mention hash map variant**: "For subarray sum, I'll use prefix sum with hash map"
- **2D case**: Explain inclusion-exclusion clearly with diagram
- **Difference array**: "This is useful when we have many range updates"

**Common follow-ups:**
- "Can you handle updates?" → Prefix sum needs rebuild (O(n)), or use Segment Tree
- "What about 2D?" → Show 2D prefix sum formula
- "Optimize space?" → Sometimes can compute on-the-fly without storing

**Red flags to avoid:**
- Forgetting prefix[0] = 0 initialization
- Wrong indices in range queries (off-by-one)
- Not handling negative numbers in subarray sum problems
- Incorrect 2D prefix sum formula (forgetting subtraction term)

---

## 6. Stack

---

### ❓ When should I use this?

- Need **Last-In-First-Out (LIFO)** behavior
- Keywords: "matching", "valid parentheses", "next greater", "nearest smaller"
- **Backtracking** through recent elements
- Problems involving **nesting** or **hierarchy**
- When you need to "remember" recent items and "forget" them in reverse order
- **Monotonic stack**: maintain increasing/decreasing order

### 🧠 Core Idea (Intuition)

Stack is like a **stack of plates**: you can only add/remove from the top.

**Three main patterns:**
1. **Matching/Validation**: Parentheses, tags, brackets
2. **Monotonic Stack**: Next greater/smaller element (keep stack sorted)
3. **State Tracking**: Store recent states, backtrack when needed

**Mental model**: 
- Like browser back button (stack of pages)
- Like undo functionality (stack of actions)
- Like nested function calls (call stack)

### 🧩 Common Problem Types

- Valid parentheses
- Next greater/smaller element
- Largest rectangle in histogram
- Trapping rain water (stack approach)
- Evaluate expressions (postfix, infix)
- Daily temperatures
- Asteroid collision
- Decode strings
- Min stack (stack with O(1) min)

### 🧱 Template (Python)

```python
# Pattern 1: Basic Stack Operations
def basic_stack():
    stack = []
    
    # Push
    stack.append(item)
    
    # Pop
    if stack:
        top = stack.pop()
    
    # Peek
    if stack:
        top = stack[-1]
    
    # Check empty
    is_empty = len(stack) == 0

# Pattern 2: Matching/Validation
def validate_with_stack(s):
    stack = []
    matching = {'(': ')', '[': ']', '{': '}'}
    
    for char in s:
        if char in matching:  # Opening bracket
            stack.append(char)
        else:  # Closing bracket
            if not stack or matching[stack.pop()] != char:
                return False
    
    return len(stack) == 0

# Pattern 3: Monotonic Stack (Next Greater Element)
def next_greater_element(arr):
    n = len(arr)
    result = [-1] * n
    stack = []  # Stores indices
    
    for i in range(n):
        # Maintain decreasing stack
        while stack and arr[stack[-1]] < arr[i]:
            idx = stack.pop()
            result[idx] = arr[i]
        stack.append(i)
    
    return result

# Pattern 4: Monotonic Stack (Next Smaller Element)
def next_smaller_element(arr):
    n = len(arr)
    result = [-1] * n
    stack = []
    
    for i in range(n):
        # Maintain increasing stack
        while stack and arr[stack[-1]] > arr[i]:
            idx = stack.pop()
            result[idx] = arr[i]
        stack.append(i)
    
    return result

# Pattern 5: Stack with Min/Max Tracking
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []  # Parallel stack tracking minimums
    
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        if self.stack:
            val = self.stack.pop()
            if val == self.min_stack[-1]:
                self.min_stack.pop()
            return val
    
    def top(self):
        return self.stack[-1] if self.stack else None
    
    def getMin(self):
        return self.min_stack[-1] if self.min_stack else None

# Pattern 6: Expression Evaluation
def evaluate_postfix(tokens):
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token not in operators:
            stack.append(int(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(int(a / b))  # Truncate toward zero
    
    return stack[0]
```

### 📌 Step-by-Step Walkthrough

**Example 1: Valid Parentheses "([{}])"**

```
Stack: []

char='(': opening → push → ['(']
char='[': opening → push → ['(', '[']
char='{': opening → push → ['(', '[', '{']
char='}': closing, matches '{' → pop → ['(', '[']
char=']': closing, matches '[' → pop → ['(']
char=')': closing, matches '(' → pop → []

Stack empty → Valid ✓
```

**Example 2: Next Greater Element [4,5,2,10]**

```
Stack stores indices, not values

i=0, val=4:
  stack=[] → push 0 → [0]
  result=[-1,-1,-1,-1]

i=1, val=5:
  arr[0]=4 < 5 → pop 0, result[0]=5
  stack=[] → push 1 → [1]
  result=[5,-1,-1,-1]

i=2, val=2:
  arr[1]=5 > 2 → don't pop
  push 2 → [1,2]
  result=[5,-1,-1,-1]

i=3, val=10:
  arr[2]=2 < 10 → pop 2, result[2]=10
  arr[1]=5 < 10 → pop 1, result[1]=10
  stack=[] → push 3 → [3]
  result=[5,10,10,-1]

Final: [5,10,10,-1]
```

**Example 3: Daily Temperatures [73,74,75,71,69,72,76,73]**

```
Find days until warmer temperature = next greater element

Stack: indices of days

i=0: push 0 → [0]
i=1: 74>73 → pop 0, answer[0]=1 → push 1 → [1]
i=2: 75>74 → pop 1, answer[1]=1 → push 2 → [2]
i=3: 71<75 → push 3 → [2,3]
i=4: 69<71 → push 4 → [2,3,4]
i=5: 72>69 → pop 4, answer[4]=1
      72>71 → pop 3, answer[3]=2
      72<75 → push 5 → [2,5]
i=6: 76>72 → pop 5, answer[5]=1
      76>75 → pop 2, answer[2]=4
      push 6 → [6]
i=7: 73<76 → push 7 → [6,7]

Answer: [1,1,4,2,1,1,0,0]
```

### 🧪 Solved Examples

**1. Valid Parentheses**
```python
def isValid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:  # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        else:  # Opening bracket
            stack.append(char)
    
    return len(stack) == 0
```

**2. Daily Temperatures**
```python
def dailyTemperatures(temperatures):
    n = len(temperatures)
    answer = [0] * n
    stack = []  # Stores indices
    
    for i in range(n):
        while stack and temperatures[stack[-1]] < temperatures[i]:
            prev_day = stack.pop()
            answer[prev_day] = i - prev_day
        stack.append(i)
    
    return answer
```

**3. Largest Rectangle in Histogram**
```python
def largestRectangleArea(heights):
    stack = []  # Stores indices
    max_area = 0
    heights.append(0)  # Sentinel to flush stack
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    
    heights.pop()  # Remove sentinel
    return max_area
```

**4. Evaluate Reverse Polish Notation**
```python
def evalRPN(tokens):
    stack = []
    
    for token in tokens:
        if token in ['+', '-', '*', '/']:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:  # Division
                stack.append(int(a / b))  # Truncate toward zero
        else:
            stack.append(int(token))
    
    return stack[0]
```

**5. Min Stack**
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
    
    def top(self):
        return self.stack[-1]
    
    def getMin(self):
        return self.min_stack[-1]
```

**6. Decode String**
```python
def decodeString(s):
    stack = []
    current_num = 0
    current_str = ""
    
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            # Push current state
            stack.append((current_str, current_num))
            current_str = ""
            current_num = 0
        elif char == ']':
            # Pop and decode
            prev_str, num = stack.pop()
            current_str = prev_str + current_str * num
        else:
            current_str += char
    
    return current_str
```

**7. Asteroid Collision**
```python
def asteroidCollision(asteroids):
    stack = []
    
    for asteroid in asteroids:
        alive = True
        
        while alive and asteroid < 0 and stack and stack[-1] > 0:
            # Collision: right-moving vs left-moving
            if stack[-1] < -asteroid:
                stack.pop()  # Top asteroid destroyed
                continue
            elif stack[-1] == -asteroid:
                stack.pop()  # Both destroyed
            alive = False  # Current asteroid destroyed
        
        if alive:
            stack.append(asteroid)
    
    return stack
```

**8. Remove K Digits**
```python
def removeKdigits(num, k):
    stack = []
    
    for digit in num:
        # Remove larger digits from stack
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    
    # Remove remaining k digits from end
    stack = stack[:-k] if k > 0 else stack
    
    # Remove leading zeros and return
    result = ''.join(stack).lstrip('0')
    return result if result else '0'
```

**9. Trapping Rain Water (Stack Approach)**
```python
def trap(height):
    stack = []  # Stores indices
    water = 0
    
    for i in range(len(height)):
        while stack and height[i] > height[stack[-1]]:
            bottom = stack.pop()
            
            if not stack:
                break
            
            distance = i - stack[-1] - 1
            bounded_height = min(height[i], height[stack[-1]]) - height[bottom]
            water += distance * bounded_height
        
        stack.append(i)
    
    return water
```

**10. Basic Calculator II**
```python
def calculate(s):
    stack = []
    num = 0
    operation = '+'
    
    for i, char in enumerate(s):
        if char.isdigit():
            num = num * 10 + int(char)
        
        if char in '+-*/' or i == len(s) - 1:
            if operation == '+':
                stack.append(num)
            elif operation == '-':
                stack.append(-num)
            elif operation == '*':
                stack.append(stack.pop() * num)
            elif operation == '/':
                stack.append(int(stack.pop() / num))
            
            operation = char
            num = 0
    
    return sum(stack)
```

### ⚠️ Edge Cases & Pitfalls

- **Empty stack pop**: Always check `if stack` before popping
- **Matching problems**: Ensure stack is empty at end
- **Monotonic stack**: Decide increasing vs decreasing based on problem
- **Index vs value**: Store indices when you need position information
- **Sentinel values**: Adding dummy elements can simplify code
- **Integer division**: Python3 `/` vs `//` (truncate toward zero vs floor)
- **Leading zeros**: In string manipulation problems
- **Overflow**: When multiplying/adding large numbers

### ⏱️ Time & Space Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Push | O(1) | - | Amortized O(1) |
| Pop | O(1) | - | |
| Peek/Top | O(1) | - | |
| Search | O(n) | - | Need to iterate |
| Valid parentheses | O(n) | O(n) | Worst case: all opening |
| Next greater element | O(n) | O(n) | Each element pushed/popped once |
| Largest rectangle | O(n) | O(n) | Amortized O(1) per element |

**Key insight**: Monotonic stack is O(n) because each element is pushed and popped at most once.

### 🧠 Interview Tips

- **Clarify input**: "Are there only these three types of brackets?"
- **Explain monotonic stack**: "I'll maintain a decreasing stack to find next greater"
- **Draw the stack**: Visualize stack state at each step
- **Mention amortized complexity**: "Although there's a while loop, it's O(n) total"
- **Edge cases**: Empty string, unmatched brackets, all same elements
- **Space optimization**: Sometimes can avoid stack with clever iteration

**Common follow-ups:**
- "Can you do it without extra space?" → Usually no for stack problems
- "What if there are multiple types of brackets?" → Extend mapping
- "Can you handle nested structures?" → Stack naturally handles nesting

**Red flags to avoid:**
- Popping from empty stack (IndexError)
- Not clearing stack state between test cases
- Confusing when to use indices vs values
- Wrong monotonic stack direction (increasing vs decreasing)

---

## 7. Queue / Deque / Monotonic Queue

---

### ❓ When should I use this?

- Need **First-In-First-Out (FIFO)** behavior
- Keywords: "level order", "BFS", "sliding window maximum", "recent requests"
- **Process in order**: Tasks, requests, nodes in a tree
- **Deque (Double-ended queue)**: Add/remove from both ends
- **Monotonic queue**: Sliding window maximum/minimum

### 🧠 Core Idea (Intuition)

**Queue**: Like a line at a store - first person in line is served first

**Deque**: Like a deck of cards - can add/remove from both ends

**Monotonic Queue**: Keep queue sorted (increasing/decreasing) by removing useless elements

**Mental model**:
- Queue: Print queue, task scheduler
- Deque: Browser history (forward/back), sliding window
- Monotonic queue: Finding max/min in sliding windows

### 🧩 Common Problem Types

- BFS (tree/graph level-order traversal)
- Sliding window maximum/minimum
- Recent counter (time-based queue)
- Design hit counter
- Moving average from data stream
- Implement stack using queues
- Rotting oranges
- Shortest path in unweighted graph

### 🧱 Template (Python)

```python
from collections import deque

# Pattern 1: Basic Queue Operations
def basic_queue():
    queue = deque()
    
    # Enqueue (add to right)
    queue.append(item)
    
    # Dequeue (remove from left)
    if queue:
        front = queue.popleft()
    
    # Peek front
    if queue:
        front = queue[0]
    
    # Check empty
    is_empty = len(queue) == 0

# Pattern 2: BFS Template
def bfs(start_node):
    queue = deque([start_node])
    visited = {start_node}
    
    while queue:
        node = queue.popleft()
        
        # Process node
        process(node)
        
        # Add neighbors
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Pattern 3: Level-Order BFS (Track Levels)
def level_order_bfs(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result

# Pattern 4: Deque for Sliding Window
def sliding_window_deque(arr, k):
    dq = deque()
    result = []
    
    for i in range(len(arr)):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Maintain order (remove useless elements)
        while dq and arr[dq[-1]] < arr[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result once window is full
        if i >= k - 1:
            result.append(arr[dq[0]])
    
    return result

# Pattern 5: Monotonic Queue (Sliding Window Maximum)
class MonotonicQueue:
    def __init__(self):
        self.dq = deque()
    
    def push(self, val):
        # Remove smaller elements (they're useless)
        while self.dq and self.dq[-1] < val:
            self.dq.pop()
        self.dq.append(val)
    
    def pop(self, val):
        # Only pop if it's the front element
        if self.dq and self.dq[0] == val:
            self.dq.popleft()
    
    def max(self):
        return self.dq[0] if self.dq else None

# Pattern 6: Time-Based Queue (Recent Counter)
class RecentCounter:
    def __init__(self):
        self.queue = deque()
    
    def ping(self, t):
        self.queue.append(t)
        
        # Remove old requests (older than 3000ms)
        while self.queue and self.queue[0] < t - 3000:
            self.queue.popleft()
        
        return len(self.queue)
```

### 📌 Step-by-Step Walkthrough

**Example 1: BFS Level Order [[3],[9,20],[15,7]]**

```
Tree:      3
         /   \
        9    20
            /  \
           15   7

Queue: [3], level=0
  Process 3, add children → Queue: [9, 20]
  Level 0: [3]

Queue: [9, 20], level=1
  Process 9, no children → Queue: [20]
  Process 20, add children → Queue: [15, 7]
  Level 1: [9, 20]

Queue: [15, 7], level=2
  Process 15, no children → Queue: [7]
  Process 7, no children → Queue: []
  Level 2: [15, 7]

Result: [[3], [9,20], [15,7]]
```

**Example 2: Sliding Window Maximum [1,3,-1,-3,5,3,6,7], k=3**

```
Deque stores indices in decreasing order of values

i=0, val=1: dq=[0]
i=1, val=3: 3>1 → remove 0 → dq=[1]
i=2, val=-1: -1<3 → dq=[1,2], window full → max=3

i=3, val=-3: -3<-1 → dq=[1,2,3], max=3
i=4, val=5: 5>-3 → remove 3
            5>-1 → remove 2
            5>3 → remove 1
            dq=[4], max=5

i=5, val=3: 3<5 → dq=[4,5], max=5
i=6, val=6: 6>3 → remove 5
            6>5 → remove 4
            dq=[6], max=6

i=7, val=7: 7>6 → remove 6
            dq=[7], max=7

Result: [3,3,5,5,6,7]
```

### 🧪 Solved Examples

**1. Binary Tree Level Order Traversal**
```python
def levelOrder(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
```

**2. Sliding Window Maximum**
```python
def maxSlidingWindow(nums, k):
    dq = deque()  # Stores indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements (useless)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

**3. Number of Recent Calls**
```python
class RecentCounter:
    def __init__(self):
        self.queue = deque()
    
    def ping(self, t):
        self.queue.append(t)
        while self.queue[0] < t - 3000:
            self.queue.popleft()
        return len(self.queue)
```

**4. Moving Average from Data Stream**
```python
class MovingAverage:
    def __init__(self, size):
        self.size = size
        self.queue = deque()
        self.window_sum = 0
    
    def next(self, val):
        self.queue.append(val)
        self.window_sum += val
        
        if len(self.queue) > self.size:
            self.window_sum -= self.queue.popleft()
        
        return self.window_sum / len(self.queue)
```

**5. Rotting Oranges (Multi-source BFS)**
```python
def orangesRotting(grid):
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0
    
    # Find all rotten oranges and count fresh ones
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))  # (row, col, time)
            elif grid[r][c] == 1:
                fresh_count += 1
    
    if fresh_count == 0:
        return 0
    
    # BFS
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    max_time = 0
    
    while queue:
        r, c, time = queue.popleft()
        max_time = max(max_time, time)
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh_count -= 1
                queue.append((nr, nc, time + 1))
    
    return max_time if fresh_count == 0 else -1
```

**6. Implement Stack Using Queues**
```python
class MyStack:
    def __init__(self):
        self.queue = deque()
    
    def push(self, x):
        self.queue.append(x)
        # Rotate queue to make last element first
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())
    
    def pop(self):
        return self.queue.popleft()
    
    def top(self):
        return self.queue[0]
    
    def empty(self):
        return len(self.queue) == 0
```

**7. Perfect Squares (BFS)**
```python
def numSquares(n):
    if n <= 0:
        return 0
    
    # BFS to find shortest path
    queue = deque([(n, 0)])  # (remaining, steps)
    visited = {n}
    
    while queue:
        remaining, steps = queue.popleft()
        
        # Try all perfect squares
        i = 1
        while i * i <= remaining:
            next_remaining = remaining - i * i
            
            if next_remaining == 0:
                return steps + 1
            
            if next_remaining not in visited:
                visited.add(next_remaining)
                queue.append((next_remaining, steps + 1))
            
            i += 1
    
    return -1
```

**8. Shortest Path in Binary Matrix**
```python
def shortestPathBinaryMatrix(grid):
    if grid[0][0] == 1 or grid[-1][-1] == 1:
        return -1
    
    n = len(grid)
    if n == 1:
        return 1
    
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    grid[0][0] = 1  # Mark visited
    
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    
    while queue:
        r, c, dist = queue.popleft()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if nr == n-1 and nc == n-1:
                return dist + 1
            
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                grid[nr][nc] = 1
                queue.append((nr, nc, dist + 1))
    
    return -1
```

### ⚠️ Edge Cases & Pitfalls

- **Empty queue**: Check before `popleft()` or accessing `queue[0]`
- **BFS visited set**: Mark as visited when adding to queue, not when processing
- **Level-order traversal**: Use `len(queue)` at start of each level
- **Deque indices**: Remember `dq[0]` is front, `dq[-1]` is back
- **Monotonic queue**: Decide increasing vs decreasing based on problem
- **Time-based queue**: Remove expired elements before counting
- **Multi-source BFS**: Add all sources to queue initially

### ⏱️ Time & Space Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Enqueue | O(1) | - | |
| Dequeue | O(1) | - | |
| Peek | O(1) | - | |
| BFS | O(V + E) | O(V) | V=vertices, E=edges |
| Level-order | O(n) | O(w) | w=max width of tree |
| Sliding window max | O(n) | O(k) | Each element added/removed once |
| Recent counter | O(1) amortized | O(w) | w=window size |

### 🧠 Interview Tips

- **BFS vs DFS**: "BFS finds shortest path, DFS explores deeply"
- **Queue choice**: "I'll use deque for O(1) operations at both ends"
- **Level tracking**: "I'll process nodes level by level using queue size"
- **Visited set**: "I mark nodes as visited when adding to queue to avoid duplicates"
- **Monotonic queue**: "I maintain decreasing order to efficiently find maximum"

**Common follow-ups:**
- "Why deque instead of list?" → O(1) popleft() vs O(n)
- "Can you track the path?" → Store parent pointers or full paths
- "What about weighted graphs?" → Need Dijkstra, not BFS

**Red flags to avoid:**
- Using list.pop(0) instead of deque.popleft() (O(n) vs O(1))
- Not marking nodes as visited (infinite loops)
- Processing same level multiple times
- Forgetting to check if queue is empty

---

## 8. Linked List (All Variants)

---

### ❓ When should I use this?

- Dynamic size collection with efficient insertion/deletion
- Keywords: "in-place", "constant space", "reverse", "cycle", "merge"
- When you need O(1) insertion/deletion at known positions
- Problems involving **pointers** and **node manipulation**
- **Fast/slow pointers** for cycle detection, middle element

### 🧠 Core Idea (Intuition)

Linked list: Chain of nodes where each node points to the next

**Key patterns:**
1. **Two pointers**: Fast/slow for cycles, middle, kth from end
2. **Reversal**: Reverse links iteratively or recursively
3. **Merge**: Combine sorted lists
4. **Dummy head**: Simplifies edge cases

**Mental model**: Like a treasure hunt where each clue points to the next location

### 🧩 Common Problem Types

- Reverse linked list
- Detect cycle / find cycle start
- Find middle element
- Merge two sorted lists
- Remove nth node from end
- Palindrome linked list
- Intersection of two lists
- Add two numbers (as linked lists)
- Copy list with random pointer
- LRU Cache

### 🧱 Template (Python)

```python
# Definition for singly-linked list
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Pattern 1: Iterative Traversal
def traverse(head):
    current = head
    while current:
        process(current)
        current = current.next

# Pattern 2: Fast/Slow Pointers
def fast_slow_pattern(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        # Check condition (e.g., cycle detection)
        if slow == fast:
            return True
    
    return False

# Pattern 3: Reverse Linked List (Iterative)
def reverse_iterative(head):
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev

# Pattern 4: Reverse Linked List (Recursive)
def reverse_recursive(head):
    if not head or not head.next:
        return head
    
    new_head = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    
    return new_head

# Pattern 5: Dummy Head Pattern
def dummy_head_pattern(head):
    dummy = ListNode(0)
    dummy.next = head
    current = dummy
    
    while current.next:
        # Manipulate current.next
        if should_remove(current.next):
            current.next = current.next.next
        else:
            current = current.next
    
    return dummy.next

# Pattern 6: Merge Two Lists
def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    current.next = l1 if l1 else l2
    return dummy.next

# Pattern 7: Find Nth Node from End
def find_nth_from_end(head, n):
    fast = slow = head
    
    # Move fast n steps ahead
    for _ in range(n):
        if not fast:
            return None
        fast = fast.next
    
    # Move both until fast reaches end
    while fast:
        fast = fast.next
        slow = slow.next
    
    return slow

# Pattern 8: Detect Cycle and Find Start
def detect_cycle(head):
    slow = fast = head
    
    # Detect cycle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle
    
    # Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow
```

### 📌 Step-by-Step Walkthrough

**Example 1: Reverse List [1→2→3→4→5]**

```
Initial: 1→2→3→4→5

Step 1: prev=None, curr=1
  next=2, 1→None, prev=1, curr=2
  Result: None←1  2→3→4→5

Step 2: prev=1, curr=2
  next=3, 2→1, prev=2, curr=3
  Result: None←1←2  3→4→5

Step 3: prev=2, curr=3
  next=4, 3→2, prev=3, curr=4
  Result: None←1←2←3  4→5

Step 4: prev=3, curr=4
  next=5, 4→3, prev=4, curr=5
  Result: None←1←2←3←4  5

Step 5: prev=4, curr=5
  next=None, 5→4, prev=5, curr=None
  Result: None←1←2←3←4←5

Return prev=5
```

**Example 2: Detect Cycle**

```
List: 1→2→3→4→5
           ↑     ↓
           8←7←6

slow and fast start at 1

Step 1: slow=2, fast=3
Step 2: slow=3, fast=5
Step 3: slow=4, fast=7
Step 4: slow=5, fast=3
Step 5: slow=6, fast=5
Step 6: slow=7, fast=7 → Cycle detected!

Find start:
  slow=1, fast=7
  slow=2, fast=8
  slow=3, fast=3 → Cycle starts at 3
```

### 🧪 Solved Examples

**1. Reverse Linked List**
```python
def reverseList(head):
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev
```

**2. Linked List Cycle**
```python
def hasCycle(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    
    return False
```

**3. Linked List Cycle II (Find Start)**
```python
def detectCycle(head):
    slow = fast = head
    
    # Detect cycle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None
    
    # Find start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow
```

**4. Middle of Linked List**
```python
def middleNode(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow
```

**5. Remove Nth Node From End**
```python
def removeNthFromEnd(head, n):
    dummy = ListNode(0)
    dummy.next = head
    fast = slow = dummy
    
    # Move fast n+1 steps ahead
    for _ in range(n + 1):
        fast = fast.next
    
    # Move both until fast reaches end
    while fast:
        fast = fast.next
        slow = slow.next
    
    # Remove nth node
    slow.next = slow.next.next
    
    return dummy.next
```

**6. Merge Two Sorted Lists**
```python
def mergeTwoLists(l1, l2):
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    current.next = l1 if l1 else l2
    return dummy.next
```

**7. Palindrome Linked List**
```python
def isPalindrome(head):
    # Find middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    prev = None
    while slow:
        next_node = slow.next
        slow.next = prev
        prev = slow
        slow = next_node
    
    # Compare both halves
    left, right = head, prev
    while right:
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    
    return True
```

**8. Intersection of Two Linked Lists**
```python
def getIntersectionNode(headA, headB):
    if not headA or not headB:
        return None
    
    pA, pB = headA, headB
    
    # When one reaches end, switch to other list's head
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA
    
    return pA  # Either intersection or None
```

**9. Add Two Numbers**
```python
def addTwoNumbers(l1, l2):
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        current.next = ListNode(total % 10)
        
        current = current.next
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
    
    return dummy.next
```

**10. Copy List with Random Pointer**
```python
def copyRandomList(head):
    if not head:
        return None
    
    # Step 1: Create copied nodes interleaved
    current = head
    while current:
        copy = Node(current.val)
        copy.next = current.next
        current.next = copy
        current = copy.next
    
    # Step 2: Set random pointers
    current = head
    while current:
        if current.random:
            current.next.random = current.random.next
        current = current.next.next
    
    # Step 3: Separate lists
    dummy = Node(0)
    copy_current = dummy
    current = head
    
    while current:
        copy_current.next = current.next
        current.next = current.next.next
        
        copy_current = copy_current.next
        current = current.next
    
    return dummy.next
```

**11. Reorder List (L0→Ln→L1→Ln-1→...)**
```python
def reorderList(head):
    if not head or not head.next:
        return
    
    # Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    second = slow.next
    slow.next = None
    prev = None
    while second:
        next_node = second.next
        second.next = prev
        prev = second
        second = next_node
    
    # Merge two halves
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first, second = tmp1, tmp2
```

### ⚠️ Edge Cases & Pitfalls

- **Null head**: Always check `if not head`
- **Single node**: Many algorithms need at least 2 nodes
- **Dummy head**: Use to simplify edge cases (removing head, merging)
- **Losing references**: Save `next` pointer before modifying `current.next`
- **Cycle detection**: Fast pointer needs `fast and fast.next` check
- **Off-by-one**: Nth from end requires careful pointer positioning
- **Memory leaks**: In languages with manual memory management
- **Modifying input**: Clarify if you can modify the original list

### ⏱️ Time & Space Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Traversal | O(n) | O(1) | |
| Reverse | O(n) | O(1) | Iterative |
| Reverse (recursive) | O(n) | O(n) | Call stack |
| Cycle detection | O(n) | O(1) | Fast/slow pointers |
| Find middle | O(n) | O(1) | Fast/slow pointers |
| Merge two lists | O(n+m) | O(1) | |
| Remove nth from end | O(n) | O(1) | Two pointers |

### 🧠 Interview Tips

- **Draw it**: Sketch nodes and pointers on whiteboard
- **Dummy head**: "I'll use a dummy head to simplify edge cases"
- **Fast/slow pointers**: "Slow moves 1 step, fast moves 2 steps"
- **Explain reversal**: Show how pointers change step by step
- **Edge cases**: Null, single node, two nodes
- **In-place**: Emphasize O(1) space when applicable

**Common follow-ups:**
- "Can you do it recursively?" → Show recursive solution
- "What if it's a doubly linked list?" → Adjust pointer logic
- "Can you do it in one pass?" → Use two pointers technique

**Red flags to avoid:**
- Losing reference to next node before reassigning
- Not checking for null pointers
- Infinite loops in cycle problems
- Forgetting to return dummy.next (not dummy)

---

## 11. Tree Traversals (DFS/BFS)

---

### ❓ When should I use this?

- Working with **tree or graph structures**
- Keywords: "traverse", "visit all nodes", "level order", "in-order", "pre-order", "post-order"
- **DFS (Depth-First Search)**: Explore deep before wide (stack/recursion)
- **BFS (Breadth-First Search)**: Explore level by level (queue)
- Need to process nodes in specific order or find shortest path

**When to choose DFS vs BFS:**
- **DFS**: Path finding, tree serialization, topological sort, backtracking
- **BFS**: Shortest path (unweighted), level-order, minimum depth

### 🧠 Core Idea (Intuition)

**DFS**: Like exploring a maze by going as far as possible down one path before backtracking

**BFS**: Like ripples in water spreading outward from a center

**Three DFS orders for binary trees:**
1. **Pre-order** (Root → Left → Right): Process before children (copying tree structure)
2. **In-order** (Left → Root → Right): BST gives sorted order
3. **Post-order** (Left → Right → Root): Process after children (deleting tree, calculating heights)

**Mental model**:
- DFS: Stack of plates (LIFO) or recursive call stack
- BFS: Queue at amusement park (FIFO)

### 🧩 Common Problem Types

**DFS:**
- All paths from root to leaf
- Maximum/minimum depth
- Path sum problems
- Validate BST
- Serialize/deserialize tree
- Lowest common ancestor

**BFS:**
- Level order traversal
- Minimum depth
- Binary tree right side view
- Zigzag level order
- Connect nodes at same level
- Check if tree is complete

### 🧱 Template (Python)

```python
# Tree Node Definition
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Pattern 1: DFS - Pre-order (Recursive)
def preorder_recursive(root):
    result = []
    
    def dfs(node):
        if not node:
            return
        
        result.append(node.val)  # Process root
        dfs(node.left)           # Process left
        dfs(node.right)          # Process right
    
    dfs(root)
    return result

# Pattern 2: DFS - In-order (Recursive)
def inorder_recursive(root):
    result = []
    
    def dfs(node):
        if not node:
            return
        
        dfs(node.left)           # Process left
        result.append(node.val)  # Process root
        dfs(node.right)          # Process right
    
    dfs(root)
    return result

# Pattern 3: DFS - Post-order (Recursive)
def postorder_recursive(root):
    result = []
    
    def dfs(node):
        if not node:
            return
        
        dfs(node.left)           # Process left
        dfs(node.right)          # Process right
        result.append(node.val)  # Process root
    
    dfs(root)
    return result

# Pattern 4: DFS - Pre-order (Iterative with Stack)
def preorder_iterative(root):
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push right first (so left is processed first)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result

# Pattern 5: DFS - In-order (Iterative)
def inorder_iterative(root):
    result = []
    stack = []
    current = root
    
    while current or stack:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process node
        current = stack.pop()
        result.append(current.val)
        
        # Move to right subtree
        current = current.right
    
    return result

# Pattern 6: DFS - Post-order (Iterative)
def postorder_iterative(root):
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push left first (so right is processed first)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    
    # Reverse to get post-order
    return result[::-1]

# Pattern 7: BFS - Level Order (Queue)
from collections import deque

def level_order(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result

# Pattern 8: DFS with Path Tracking
def dfs_with_path(root):
    paths = []
    
    def dfs(node, path):
        if not node:
            return
        
        path.append(node.val)
        
        # Leaf node - save path
        if not node.left and not node.right:
            paths.append(path[:])  # Make copy
        
        dfs(node.left, path)
        dfs(node.right, path)
        
        path.pop()  # Backtrack
    
    dfs(root, [])
    return paths

# Pattern 9: DFS with Return Value (Height/Depth)
def max_depth(root):
    if not root:
        return 0
    
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    return 1 + max(left_depth, right_depth)

# Pattern 10: BFS for Shortest Path/Minimum
def min_depth(root):
    if not root:
        return 0
    
    queue = deque([(root, 1)])  # (node, depth)
    
    while queue:
        node, depth = queue.popleft()
        
        # First leaf found is at minimum depth
        if not node.left and not node.right:
            return depth
        
        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))
    
    return 0
```

### 📌 Step-by-Step Walkthrough

**Example Tree:**
```
       1
      / \
     2   3
    / \
   4   5
```

**Pre-order (Root → Left → Right): [1,2,4,5,3]**
```
Visit 1 → result=[1]
  Visit 2 → result=[1,2]
    Visit 4 → result=[1,2,4]
    Visit 5 → result=[1,2,4,5]
  Visit 3 → result=[1,2,4,5,3]
```

**In-order (Left → Root → Right): [4,2,5,1,3]**
```
Go left to 4 → result=[4]
Visit 2 → result=[4,2]
Go right to 5 → result=[4,2,5]
Visit 1 → result=[4,2,5,1]
Visit 3 → result=[4,2,5,1,3]
```

**Post-order (Left → Right → Root): [4,5,2,3,1]**
```
Visit 4 → result=[4]
Visit 5 → result=[4,5]
Visit 2 → result=[4,5,2]
Visit 3 → result=[4,5,2,3]
Visit 1 → result=[4,5,2,3,1]
```

**Level-order (BFS): [[1],[2,3],[4,5]]**
```
Level 0: queue=[1] → process 1, add 2,3
Level 1: queue=[2,3] → process 2,3, add 4,5
Level 2: queue=[4,5] → process 4,5
```

### 🧪 Solved Examples

**1. Binary Tree Preorder Traversal**
```python
def preorderTraversal(root):
    result = []
    
    def dfs(node):
        if not node:
            return
        result.append(node.val)
        dfs(node.left)
        dfs(node.right)
    
    dfs(root)
    return result
```

**2. Binary Tree Level Order Traversal**
```python
def levelOrder(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
```

**3. Maximum Depth of Binary Tree**
```python
def maxDepth(root):
    if not root:
        return 0
    
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

**4. Binary Tree Paths**
```python
def binaryTreePaths(root):
    if not root:
        return []
    
    paths = []
    
    def dfs(node, path):
        if not node:
            return
        
        path.append(str(node.val))
        
        if not node.left and not node.right:
            paths.append('->'.join(path))
        else:
            dfs(node.left, path)
            dfs(node.right, path)
        
        path.pop()
    
    dfs(root, [])
    return paths
```

**5. Path Sum**
```python
def hasPathSum(root, targetSum):
    if not root:
        return False
    
    if not root.left and not root.right:
        return root.val == targetSum
    
    remaining = targetSum - root.val
    return (hasPathSum(root.left, remaining) or 
            hasPathSum(root.right, remaining))
```

**6. Binary Tree Right Side View**
```python
def rightSideView(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # Last node in level
            if i == level_size - 1:
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result
```

**7. Binary Tree Zigzag Level Order**
```python
def zigzagLevelOrder(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        current_level = deque()
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if left_to_right:
                current_level.append(node.val)
            else:
                current_level.appendleft(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(list(current_level))
        left_to_right = not left_to_right
    
    return result
```

**8. Serialize and Deserialize Binary Tree**
```python
def serialize(root):
    def dfs(node):
        if not node:
            return 'null,'
        return str(node.val) + ',' + dfs(node.left) + dfs(node.right)
    
    return dfs(root)

def deserialize(data):
    def dfs(nodes):
        val = next(nodes)
        if val == 'null':
            return None
        node = TreeNode(int(val))
        node.left = dfs(nodes)
        node.right = dfs(nodes)
        return node
    
    return dfs(iter(data.split(',')))
```

**9. Lowest Common Ancestor**
```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    
    if left and right:
        return root
    
    return left if left else right
```

**10. Diameter of Binary Tree**
```python
def diameterOfBinaryTree(root):
    diameter = 0
    
    def height(node):
        nonlocal diameter
        if not node:
            return 0
        
        left = height(node.left)
        right = height(node.right)
        
        diameter = max(diameter, left + right)
        
        return 1 + max(left, right)
    
    height(root)
    return diameter
```

### ⚠️ Edge Cases & Pitfalls

- **Null root**: Always check `if not root`
- **Single node**: Tree with only root
- **Skewed tree**: All nodes on one side (linked list-like)
- **Stack overflow**: Very deep recursion in skewed tree
- **Modifying during traversal**: Be careful with in-place modifications
- **Path tracking**: Remember to make copies of lists
- **Level tracking**: Use queue size at start of each level
- **Iterative DFS**: Different stack patterns for pre/in/post order

### ⏱️ Time & Space Complexity

| Traversal | Time | Space (Recursive) | Space (Iterative) |
|-----------|------|-------------------|-------------------|
| Pre-order | O(n) | O(h) | O(h) |
| In-order | O(n) | O(h) | O(h) |
| Post-order | O(n) | O(h) | O(h) |
| Level-order (BFS) | O(n) | - | O(w) |

Where:
- n = number of nodes
- h = height of tree (O(log n) balanced, O(n) skewed)
- w = maximum width of tree

### 🧠 Interview Tips

- **Clarify traversal type**: "Should I use pre-order, in-order, or post-order?"
- **Recursive vs iterative**: "I'll start with recursive, but can do iterative if needed"
- **Explain order**: "Pre-order processes root before children, useful for copying structure"
- **BFS for levels**: "I'll use BFS with a queue to process level by level"
- **Space complexity**: Mention call stack for recursion
- **Edge cases**: Empty tree, single node, skewed tree

**Common follow-ups:**
- "Can you do it iteratively?" → Use stack for DFS, queue for BFS
- "What if tree is very deep?" → Iterative to avoid stack overflow
- "Track parent pointers?" → Add parent tracking in traversal

**Red flags to avoid:**
- Not checking for null nodes
- Forgetting to track level in BFS
- Wrong order in DFS (pre/in/post)
- Not making copies when tracking paths

---

## 12. Binary Tree Patterns

---

### ❓ When should I use this?

- Problems involving **tree properties** or **relationships**
- Keywords: "balanced", "symmetric", "diameter", "path sum", "ancestor"
- Need to **combine information** from left and right subtrees
- **Top-down** (pass info down) vs **Bottom-up** (return info up)

### 🧠 Core Idea (Intuition)

**Top-down approach**: Pass information from parent to children (like pre-order)
- Example: Check if path sum exists with target

**Bottom-up approach**: Gather information from children and return to parent (like post-order)
- Example: Calculate height, check if balanced

**Key insight**: Most tree problems can be solved by:
1. Solving for left subtree
2. Solving for right subtree
3. Combining results

**Mental model**: Think of tree as recursive structure - solve for subtrees, then combine

### 🧩 Common Problem Types

- Check if balanced
- Check if symmetric
- Diameter of tree
- Maximum path sum
- Lowest common ancestor
- Invert/flip tree
- Flatten tree to linked list
- Count complete tree nodes
- Construct tree from traversals
- Validate BST

### 🧱 Template (Python)

```python
# Pattern 1: Top-Down (Pass Info Down)
def top_down(root, params):
    if not root:
        return base_case
    
    # Use params from parent
    result = process(root, params)
    
    # Pass updated params to children
    left_result = top_down(root.left, updated_params)
    right_result = top_down(root.right, updated_params)
    
    return combine(result, left_result, right_result)

# Pattern 2: Bottom-Up (Return Info Up)
def bottom_up(root):
    if not root:
        return base_case
    
    # Get info from children
    left_info = bottom_up(root.left)
    right_info = bottom_up(root.right)
    
    # Process current node with children's info
    current_info = process(root, left_info, right_info)
    
    return current_info

# Pattern 3: Check Property (Returns Boolean + Info)
def check_property(root):
    def helper(node):
        if not node:
            return True, base_info
        
        left_valid, left_info = helper(node.left)
        right_valid, right_info = helper(node.right)
        
        # Check if current subtree satisfies property
        current_valid = (left_valid and right_valid and 
                        check_condition(node, left_info, right_info))
        current_info = compute_info(node, left_info, right_info)
        
        return current_valid, current_info
    
    valid, _ = helper(root)
    return valid

# Pattern 4: Global Variable Pattern (For Max/Min)
def global_variable_pattern(root):
    result = float('-inf')  # or 0, depending on problem
    
    def dfs(node):
        nonlocal result
        if not node:
            return base_case
        
        left = dfs(node.left)
        right = dfs(node.right)
        
        # Update global result
        result = max(result, compute_result(node, left, right))
        
        # Return value for parent
        return compute_for_parent(node, left, right)
    
    dfs(root)
    return result

# Pattern 5: Path Tracking
def path_tracking(root, target):
    def dfs(node, current_path):
        if not node:
            return
        
        current_path.append(node.val)
        
        # Check if leaf and condition met
        if not node.left and not node.right:
            if check_condition(current_path, target):
                result.append(current_path[:])
        
        dfs(node.left, current_path)
        dfs(node.right, current_path)
        
        current_path.pop()  # Backtrack
    
    result = []
    dfs(root, [])
    return result
```

### 📌 Step-by-Step Walkthrough

**Example: Check if Balanced (Bottom-up)**

```
Tree:     1
         / \
        2   3
       / \
      4   5

Check balanced (height difference ≤ 1):

helper(4): leaf → return (True, 1)
helper(5): leaf → return (True, 1)
helper(2):
  left=(True, 1), right=(True, 1)
  |1-1| ≤ 1 ✓
  return (True, 2)

helper(3): leaf → return (True, 1)

helper(1):
  left=(True, 2), right=(True, 1)
  |2-1| ≤ 1 ✓
  return (True, 3)

Result: True (balanced)
```

**Example: Maximum Path Sum (Global Variable)**

```
Tree:     -10
          / \
         9  20
           /  \
          15   7

max_sum = -inf

dfs(9): return 9, max_sum = max(-inf, 9) = 9
dfs(15): return 15, max_sum = max(9, 15) = 15
dfs(7): return 7, max_sum = 15
dfs(20):
  left=15, right=7
  path through 20: 15+20+7=42
  max_sum = max(15, 42) = 42
  return 20+max(15,7) = 35
dfs(-10):
  left=9, right=35
  path through -10: 9+(-10)+35=34
  max_sum = max(42, 34) = 42
  return -10+max(9,35) = 25

Result: 42
```

### 🧪 Solved Examples

**1. Balanced Binary Tree**
```python
def isBalanced(root):
    def height(node):
        if not node:
            return 0
        
        left = height(node.left)
        if left == -1:
            return -1
        
        right = height(node.right)
        if right == -1:
            return -1
        
        if abs(left - right) > 1:
            return -1
        
        return 1 + max(left, right)
    
    return height(root) != -1
```

**2. Symmetric Tree**
```python
def isSymmetric(root):
    def is_mirror(left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        
        return (left.val == right.val and
                is_mirror(left.left, right.right) and
                is_mirror(left.right, right.left))
    
    return is_mirror(root, root)
```

**3. Diameter of Binary Tree**
```python
def diameterOfBinaryTree(root):
    diameter = 0
    
    def height(node):
        nonlocal diameter
        if not node:
            return 0
        
        left = height(node.left)
        right = height(node.right)
        
        diameter = max(diameter, left + right)
        
        return 1 + max(left, right)
    
    height(root)
    return diameter
```

**4. Binary Tree Maximum Path Sum**
```python
def maxPathSum(root):
    max_sum = float('-inf')
    
    def max_gain(node):
        nonlocal max_sum
        if not node:
            return 0
        
        # Max gain from left and right (ignore negative)
        left = max(max_gain(node.left), 0)
        right = max(max_gain(node.right), 0)
        
        # Path through current node
        current_path = node.val + left + right
        max_sum = max(max_sum, current_path)
        
        # Return max gain if continue to parent
        return node.val + max(left, right)
    
    max_gain(root)
    return max_sum
```

**5. Invert Binary Tree**
```python
def invertTree(root):
    if not root:
        return None
    
    # Swap children
    root.left, root.right = root.right, root.left
    
    # Recursively invert subtrees
    invertTree(root.left)
    invertTree(root.right)
    
    return root
```

**6. Subtree of Another Tree**
```python
def isSubtree(root, subRoot):
    def is_same(p, q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        return (p.val == q.val and
                is_same(p.left, q.left) and
                is_same(p.right, q.right))
    
    if not root:
        return False
    
    if is_same(root, subRoot):
        return True
    
    return (isSubtree(root.left, subRoot) or
            isSubtree(root.right, subRoot))
```

**7. Construct Binary Tree from Preorder and Inorder**
```python
def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
    
    root_val = preorder[0]
    root = TreeNode(root_val)
    
    mid = inorder.index(root_val)
    
    root.left = buildTree(preorder[1:mid+1], inorder[:mid])
    root.right = buildTree(preorder[mid+1:], inorder[mid+1:])
    
    return root
```

**8. Flatten Binary Tree to Linked List**
```python
def flatten(root):
    def flatten_helper(node):
        if not node:
            return None
        
        # Flatten left and right
        left_tail = flatten_helper(node.left)
        right_tail = flatten_helper(node.right)
        
        # If there's a left subtree, insert it between node and right
        if left_tail:
            left_tail.right = node.right
            node.right = node.left
            node.left = None
        
        # Return the tail of the flattened tree
        return right_tail or left_tail or node
    
    flatten_helper(root)
```

**9. Count Complete Tree Nodes**
```python
def countNodes(root):
    if not root:
        return 0
    
    def get_height(node):
        height = 0
        while node:
            height += 1
            node = node.left
        return height
    
    left_height = get_height(root.left)
    right_height = get_height(root.right)
    
    if left_height == right_height:
        # Left subtree is perfect
        return (1 << left_height) + countNodes(root.right)
    else:
        # Right subtree is perfect
        return (1 << right_height) + countNodes(root.left)
```

**10. Path Sum II (All Paths)**
```python
def pathSum(root, targetSum):
    result = []
    
    def dfs(node, current_sum, path):
        if not node:
            return
        
        path.append(node.val)
        current_sum += node.val
        
        if not node.left and not node.right and current_sum == targetSum:
            result.append(path[:])
        
        dfs(node.left, current_sum, path)
        dfs(node.right, current_sum, path)
        
        path.pop()
    
    dfs(root, 0, [])
    return result
```

### ⚠️ Edge Cases & Pitfalls

- **Null nodes**: Always check `if not node`
- **Single node tree**: Edge case for many algorithms
- **Negative values**: Path sum problems may have negative node values
- **Integer overflow**: Sum calculations with large values
- **Modifying tree structure**: Be careful with in-place modifications
- **Global variables**: Remember to use `nonlocal` in Python
- **Path copying**: Make copies when storing paths
- **Height vs depth**: Height from bottom, depth from top

### ⏱️ Time & Space Complexity

| Problem | Time | Space | Notes |
|---------|------|-------|-------|
| Check balanced | O(n) | O(h) | Visit each node once |
| Symmetric tree | O(n) | O(h) | Compare all nodes |
| Diameter | O(n) | O(h) | Single traversal |
| Max path sum | O(n) | O(h) | Visit each node once |
| Invert tree | O(n) | O(h) | Process each node |
| Construct from traversals | O(n²) or O(n) | O(n) | With hashmap: O(n) |
| Count complete tree | O(log²n) | O(log n) | Optimized for complete tree |

### 🧠 Interview Tips

- **Choose approach**: "I'll use bottom-up to gather info from children"
- **Explain recursion**: "Base case is null node, then combine left and right results"
- **Global variables**: "I'll use a variable to track the maximum across all paths"
- **Edge cases**: Empty tree, single node, all same values
- **Optimization**: "For complete tree, I can optimize using binary search"
- **Draw tree**: Sketch example and trace through algorithm

**Common follow-ups:**
- "Can you do it iteratively?" → Usually harder, may need stack
- "What if tree is very large?" → Discuss space optimization
- "Handle null values?" → Clarify if nulls are allowed

**Red flags to avoid:**
- Not handling null nodes
- Forgetting to return values in recursion
- Modifying tree when problem asks not to
- Not using nonlocal for global variables in Python

---

## 13. Binary Search Tree (BST)

---

### ❓ When should I use this?

- Tree where **left < root < right** property holds
- Keywords: "binary search tree", "sorted", "find k-th smallest", "validate BST"
- Need **O(log n) search/insert/delete** in balanced BST
- **In-order traversal gives sorted sequence**

### 🧠 Core Idea (Intuition)

BST property: For every node, all values in left subtree < node.val < all values in right subtree

**Key insights:**
- In-order traversal → sorted array
- Search is like binary search (go left if smaller, right if larger)
- Insert: Find position using BST property
- Delete: Three cases (leaf, one child, two children)

**Mental model**: Like a sorted dictionary where you can quickly find, add, or remove entries

### 🧩 Common Problem Types

- Validate BST
- Search in BST
- Insert into BST
- Delete node in BST
- K-th smallest element
- Lowest common ancestor in BST
- Convert sorted array to BST
- Range sum of BST
- Trim BST
- Inorder successor/predecessor

### 🧱 Template (Python)

```python
# Pattern 1: Search in BST
def search_bst(root, val):
    if not root:
        return None
    
    if root.val == val:
        return root
    elif val < root.val:
        return search_bst(root.left, val)
    else:
        return search_bst(root.right, val)

# Pattern 2: Insert into BST
def insert_bst(root, val):
    if not root:
        return TreeNode(val)
    
    if val < root.val:
        root.left = insert_bst(root.left, val)
    else:
        root.right = insert_bst(root.right, val)
    
    return root

# Pattern 3: Delete from BST
def delete_bst(root, key):
    if not root:
        return None
    
    if key < root.val:
        root.left = delete_bst(root.left, key)
    elif key > root.val:
        root.right = delete_bst(root.right, key)
    else:
        # Node found - three cases
        
        # Case 1: Leaf or one child
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        
        # Case 2: Two children
        # Find inorder successor (smallest in right subtree)
        min_node = find_min(root.right)
        root.val = min_node.val
        root.right = delete_bst(root.right, min_node.val)
    
    return root

def find_min(node):
    while node.left:
        node = node.left
    return node

# Pattern 4: Validate BST
def is_valid_bst(root):
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        if not (min_val < node.val < max_val):
            return False
        
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))

# Pattern 5: K-th Smallest (In-order)
def kth_smallest(root, k):
    def inorder(node):
        if not node:
            return None
        
        # Check left subtree
        left = inorder(node.left)
        if left is not None:
            return left
        
        # Check current node
        nonlocal count
        count += 1
        if count == k:
            return node.val
        
        # Check right subtree
        return inorder(node.right)
    
    count = 0
    return inorder(root)

# Pattern 6: Range Sum BST
def range_sum_bst(root, low, high):
    if not root:
        return 0
    
    total = 0
    
    # Add current node if in range
    if low <= root.val <= high:
        total += root.val
    
    # Recursively search left if possible
    if root.val > low:
        total += range_sum_bst(root.left, low, high)
    
    # Recursively search right if possible
    if root.val < high:
        total += range_sum_bst(root.right, low, high)
    
    return total

# Pattern 7: Lowest Common Ancestor in BST
def lca_bst(root, p, q):
    if not root:
        return None
    
    # Both in left subtree
    if p.val < root.val and q.val < root.val:
        return lca_bst(root.left, p, q)
    
    # Both in right subtree
    if p.val > root.val and q.val > root.val:
        return lca_bst(root.right, p, q)
    
    # Split point or one of them is root
    return root

# Pattern 8: Convert Sorted Array to BST
def sorted_array_to_bst(nums):
    if not nums:
        return None
    
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid+1:])
    
    return root
```

### 📌 Step-by-Step Walkthrough

**Example: Insert 4 into BST**

```
Initial BST:
       5
      / \
     3   7
    /
   2

Insert 4:
Start at 5: 4 < 5, go left
At 3: 4 > 3, go right
At None: Create node(4)

Result:
       5
      / \
     3   7
    / \
   2   4
```

**Example: Delete 3 (node with two children)**

```
BST:
       5
      / \
     3   7
    / \
   2   4

Delete 3 (has two children):
1. Find inorder successor: smallest in right subtree = 4
2. Replace 3 with 4
3. Delete original 4

Result:
       5
      / \
     4   7
    /
   2
```

**Example: Validate BST**

```
Tree:
       5
      / \
     3   7
    / \
   1   6

validate(5, -inf, inf):
  validate(3, -inf, 5):
    validate(1, -inf, 3): ✓
    validate(6, 3, 5): 6 > 5 ✗

Invalid BST! (6 should be < 5)
```

### 🧪 Solved Examples

**1. Search in BST**
```python
def searchBST(root, val):
    if not root or root.val == val:
        return root
    
    if val < root.val:
        return searchBST(root.left, val)
    return searchBST(root.right, val)
```

**2. Insert into BST**
```python
def insertIntoBST(root, val):
    if not root:
        return TreeNode(val)
    
    if val < root.val:
        root.left = insertIntoBST(root.left, val)
    else:
        root.right = insertIntoBST(root.right, val)
    
    return root
```

**3. Delete Node in BST**
```python
def deleteNode(root, key):
    if not root:
        return None
    
    if key < root.val:
        root.left = deleteNode(root.left, key)
    elif key > root.val:
        root.right = deleteNode(root.right, key)
    else:
        # Node found
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        
        # Two children: find min in right subtree
        min_node = root.right
        while min_node.left:
            min_node = min_node.left
        
        root.val = min_node.val
        root.right = deleteNode(root.right, root.val)
    
    return root
```

**4. Validate Binary Search Tree**
```python
def isValidBST(root):
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        if not (min_val < node.val < max_val):
            return False
        
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))
```

**5. Kth Smallest Element in BST**
```python
def kthSmallest(root, k):
    stack = []
    current = root
    
    while True:
        while current:
            stack.append(current)
            current = current.left
        
        current = stack.pop()
        k -= 1
        if k == 0:
            return current.val
        
        current = current.right
```

**6. Lowest Common Ancestor of BST**
```python
def lowestCommonAncestor(root, p, q):
    if p.val < root.val and q.val < root.val:
        return lowestCommonAncestor(root.left, p, q)
    
    if p.val > root.val and q.val > root.val:
        return lowestCommonAncestor(root.right, p, q)
    
    return root
```

**7. Convert Sorted Array to BST**
```python
def sortedArrayToBST(nums):
    if not nums:
        return None
    
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[:mid])
    root.right = sortedArrayToBST(nums[mid+1:])
    
    return root
```

**8. Range Sum of BST**
```python
def rangeSumBST(root, low, high):
    if not root:
        return 0
    
    total = 0
    if low <= root.val <= high:
        total += root.val
    
    if root.val > low:
        total += rangeSumBST(root.left, low, high)
    if root.val < high:
        total += rangeSumBST(root.right, low, high)
    
    return total
```

**9. Trim a Binary Search Tree**
```python
def trimBST(root, low, high):
    if not root:
        return None
    
    if root.val < low:
        return trimBST(root.right, low, high)
    if root.val > high:
        return trimBST(root.left, low, high)
    
    root.left = trimBST(root.left, low, high)
    root.right = trimBST(root.right, low, high)
    
    return root
```

**10. Inorder Successor in BST**
```python
def inorderSuccessor(root, p):
    successor = None
    
    while root:
        if p.val < root.val:
            successor = root
            root = root.left
        else:
            root = root.right
    
    return successor
```

### ⚠️ Edge Cases & Pitfalls

- **Not a valid BST**: Always validate BST property with min/max bounds
- **Duplicate values**: Clarify if duplicates allowed (usually go right)
- **Single node**: Edge case for delete operation
- **Empty tree**: Check `if not root`
- **Integer overflow**: When comparing with infinity
- **In-order traversal**: Remember it gives sorted order
- **Delete with two children**: Must find successor/predecessor correctly

### ⏱️ Time & Space Complexity

| Operation | Average | Worst (Skewed) | Notes |
|-----------|---------|----------------|-------|
| Search | O(log n) | O(n) | Balanced vs skewed |
| Insert | O(log n) | O(n) | |
| Delete | O(log n) | O(n) | |
| Validate | O(n) | O(n) | Must visit all nodes |
| Kth smallest | O(k) | O(n) | Early termination |
| Range sum | O(n) | O(n) | May visit all nodes |
| Convert sorted array | O(n) | O(n) | Each node created once |

Space: O(h) for recursion stack, where h = height

### 🧠 Interview Tips

- **BST property**: "I'll use the BST property: left < root < right"
- **Validate carefully**: "I need to track min/max bounds, not just compare with parent"
- **In-order gives sorted**: "In-order traversal of BST gives elements in sorted order"
- **Delete complexity**: "Delete has three cases: leaf, one child, two children"
- **Optimization**: "I can prune branches that don't contain values in range"
- **Balanced vs skewed**: Mention difference in complexity

**Common follow-ups:**
- "What if BST is not balanced?" → Discuss AVL/Red-Black trees
- "Can you do it iteratively?" → Use stack for most operations
- "Find successor/predecessor?" → Show iterative approach

**Red flags to avoid:**
- Validating BST by only comparing with parent
- Not handling all three delete cases
- Forgetting that in-order gives sorted sequence
- Not pruning in range queries

---

## 14. Heap / Priority Queue

---

### ❓ When should I use this?

- Need to repeatedly find **minimum or maximum** element
- Keywords: "k largest", "k smallest", "top k", "median", "merge k sorted"
- **Priority-based processing**: Always process most important element next
- When you need **O(log n) insert** and **O(1) peek** for min/max
- Problems involving **streaming data** or **online algorithms**

### 🧠 Core Idea (Intuition)

**Heap**: Complete binary tree where parent is always smaller (min-heap) or larger (max-heap) than children

**Priority Queue**: Abstract data structure implemented using heap

**Key properties:**
- **Min-heap**: Root is minimum element
- **Max-heap**: Root is maximum element
- Insert: O(log n), Extract min/max: O(log n), Peek: O(1)

**Mental model**: 
- Like a tournament bracket where winner (min/max) bubbles to top
- Like a hospital ER: most critical patients seen first (priority)

### 🧩 Common Problem Types

- K-th largest/smallest element
- Top K frequent elements
- Merge K sorted lists
- Find median from data stream
- Meeting rooms II (minimum rooms needed)
- Task scheduler
- Ugly number II
- K closest points to origin
- Sliding window median
- IPO (maximize capital)

### 🧱 Template (Python)

```python
import heapq

# Pattern 1: Min Heap (Default in Python)
def min_heap_operations():
    heap = []
    
    # Insert
    heapq.heappush(heap, value)
    
    # Extract min
    min_val = heapq.heappop(heap)
    
    # Peek min
    min_val = heap[0]
    
    # Heapify existing list
    heapq.heapify(arr)
    
    # Push and pop (efficient)
    heapq.heappushpop(heap, value)  # Push then pop
    heapq.heapreplace(heap, value)  # Pop then push

# Pattern 2: Max Heap (Negate Values)
def max_heap_operations():
    max_heap = []
    
    # Insert (negate for max heap)
    heapq.heappush(max_heap, -value)
    
    # Extract max
    max_val = -heapq.heappop(max_heap)
    
    # Peek max
    max_val = -max_heap[0]

# Pattern 3: K Largest Elements (Min Heap of Size K)
def k_largest(nums, k):
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)  # Remove smallest
    
    return list(heap)  # or heap[0] for k-th largest

# Pattern 4: K Smallest Elements (Max Heap of Size K)
def k_smallest(nums, k):
    heap = []
    
    for num in nums:
        heapq.heappush(heap, -num)  # Max heap
        if len(heap) > k:
            heapq.heappop(heap)  # Remove largest
    
    return [-x for x in heap]

# Pattern 5: Merge K Sorted Lists/Arrays
def merge_k_sorted(lists):
    heap = []
    result = []
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))  # (value, list_idx, elem_idx)
    
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from same list
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
    return result

# Pattern 6: Two Heaps (Median)
class MedianFinder:
    def __init__(self):
        self.small = []  # Max heap (negated)
        self.large = []  # Min heap
    
    def addNum(self, num):
        # Add to max heap (small)
        heapq.heappush(self.small, -num)
        
        # Balance: move largest from small to large
        heapq.heappush(self.large, -heapq.heappop(self.small))
        
        # Keep small size >= large size
        if len(self.small) < len(self.large):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0

# Pattern 7: Heap with Custom Comparator (Tuples)
def custom_heap():
    heap = []
    
    # Tuples are compared element by element
    # (priority, value) or (priority, secondary_priority, value)
    heapq.heappush(heap, (priority, value))
    
    # For objects, use tuple with comparison key
    heapq.heappush(heap, (obj.priority, obj.id, obj))

# Pattern 8: Fixed Size Heap (Sliding Window)
def sliding_window_heap(nums, k):
    heap = []
    result = []
    
    for i, num in enumerate(nums):
        heapq.heappush(heap, (num, i))
        
        # Remove elements outside window
        while heap and heap[0][1] <= i - k:
            heapq.heappop(heap)
        
        if i >= k - 1:
            result.append(heap[0][0])
    
    return result
```

### 📌 Step-by-Step Walkthrough

**Example 1: Find K-th Largest Element in [3,2,1,5,6,4], k=2**

```
Using min heap of size k=2:

num=3: heap=[3]
num=2: heap=[2,3]
num=1: heap=[1,2,3], size>2 → pop(1) → heap=[2,3]
num=5: heap=[2,3,5], size>2 → pop(2) → heap=[3,5]
num=6: heap=[3,5,6], size>2 → pop(3) → heap=[5,6]
num=4: heap=[4,5,6], size>2 → pop(4) → heap=[5,6]

K-th largest = heap[0] = 5
```

**Example 2: Merge K Sorted Lists [[1,4,5],[1,3,4],[2,6]]**

```
Initialize heap with first elements:
heap = [(1,0,0), (1,1,0), (2,2,0)]

Step 1: Pop (1,0,0), add 1 to result
  Next from list 0: (4,0,1)
  heap = [(1,1,0), (2,2,0), (4,0,1)]

Step 2: Pop (1,1,0), add 1 to result
  Next from list 1: (3,1,1)
  heap = [(2,2,0), (3,1,1), (4,0,1)]

Step 3: Pop (2,2,0), add 2 to result
  Next from list 2: (6,2,1)
  heap = [(3,1,1), (4,0,1), (6,2,1)]

Continue until heap empty...
Result: [1,1,2,3,4,4,5,6]
```

**Example 3: Find Median from Stream [5,15,1,3]**

```
small (max heap)  |  large (min heap)

Add 5:
  small=[-5], large=[]
  Balance: move to large
  small=[], large=[5]
  Rebalance: move back
  small=[-5], large=[]
  Median = 5

Add 15:
  small=[-5,-15], large=[]
  Balance: move to large
  small=[-5], large=[15]
  Median = (5+15)/2 = 10

Add 1:
  small=[-5,-1], large=[15]
  Balance: move to large
  small=[-1], large=[5,15]
  Rebalance: move back
  small=[-5,-1], large=[15]
  Median = 5

Add 3:
  small=[-5,-3,-1], large=[15]
  Balance: move to large
  small=[-5,-1], large=[3,15]
  Median = (5+3)/2 = 4
```

### 🧪 Solved Examples

**1. Kth Largest Element in Array**
```python
def findKthLargest(nums, k):
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]
```

**2. Top K Frequent Elements**
```python
def topKFrequent(nums, k):
    from collections import Counter
    
    count = Counter(nums)
    
    # Min heap of size k
    heap = []
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]
```

**3. Merge K Sorted Lists**
```python
def mergeKLists(lists):
    heap = []
    
    # Initialize with head of each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return dummy.next
```

**4. Find Median from Data Stream**
```python
class MedianFinder:
    def __init__(self):
        self.small = []  # Max heap
        self.large = []  # Min heap
    
    def addNum(self, num):
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        
        if len(self.small) < len(self.large):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0
```

**5. K Closest Points to Origin**
```python
def kClosest(points, k):
    heap = []
    
    for x, y in points:
        dist = x*x + y*y
        heapq.heappush(heap, (-dist, x, y))  # Max heap
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [[x, y] for dist, x, y in heap]
```

**6. Meeting Rooms II (Minimum Rooms)**
```python
def minMeetingRooms(intervals):
    if not intervals:
        return 0
    
    intervals.sort(key=lambda x: x[0])
    heap = []  # End times
    
    for start, end in intervals:
        # If earliest ending meeting is done, reuse room
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        
        heapq.heappush(heap, end)
    
    return len(heap)
```

**7. Task Scheduler**
```python
def leastInterval(tasks, n):
    from collections import Counter
    
    count = Counter(tasks)
    max_heap = [-freq for freq in count.values()]
    heapq.heapify(max_heap)
    
    time = 0
    
    while max_heap:
        temp = []
        
        # Process n+1 tasks (one cycle)
        for _ in range(n + 1):
            if max_heap:
                freq = heapq.heappop(max_heap)
                if freq + 1 < 0:  # Still has tasks
                    temp.append(freq + 1)
            time += 1
            
            if not max_heap and not temp:
                break
        
        # Put back remaining tasks
        for freq in temp:
            heapq.heappush(max_heap, freq)
    
    return time
```

**8. Ugly Number II**
```python
def nthUglyNumber(n):
    heap = [1]
    seen = {1}
    factors = [2, 3, 5]
    
    for _ in range(n):
        num = heapq.heappop(heap)
        
        for factor in factors:
            new_num = num * factor
            if new_num not in seen:
                seen.add(new_num)
                heapq.heappush(heap, new_num)
    
    return num
```

**9. Kth Smallest Element in Sorted Matrix**
```python
def kthSmallest(matrix, k):
    n = len(matrix)
    heap = []
    
    # Add first element from each row
    for r in range(min(k, n)):
        heapq.heappush(heap, (matrix[r][0], r, 0))
    
    result = 0
    for _ in range(k):
        result, r, c = heapq.heappop(heap)
        
        if c + 1 < n:
            heapq.heappush(heap, (matrix[r][c + 1], r, c + 1))
    
    return result
```

**10. IPO (Maximize Capital)**
```python
def findMaximizedCapital(k, w, profits, capital):
    projects = sorted(zip(capital, profits))
    available = []  # Max heap of profits
    i = 0
    
    for _ in range(k):
        # Add all affordable projects
        while i < len(projects) and projects[i][0] <= w:
            heapq.heappush(available, -projects[i][1])
            i += 1
        
        if not available:
            break
        
        # Take most profitable project
        w += -heapq.heappop(available)
    
    return w
```

### ⚠️ Edge Cases & Pitfalls

- **Max heap in Python**: Must negate values (no built-in max heap)
- **Empty heap**: Check `if heap` before accessing `heap[0]`
- **Duplicate values**: Heaps allow duplicates
- **Stability**: Heap is not stable (equal elements may change order)
- **Custom objects**: Use tuples for comparison or implement `__lt__`
- **Heapify**: O(n) to build heap from list, faster than n insertions
- **Two heaps for median**: Must maintain size invariant (small >= large)
- **Memory**: Heap stores all elements, can be O(n) space

### ⏱️ Time & Space Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Insert (push) | O(log n) | - | Bubble up |
| Extract min/max (pop) | O(log n) | - | Bubble down |
| Peek (top) | O(1) | - | Access root |
| Heapify | O(n) | - | Build heap from array |
| K largest/smallest | O(n log k) | O(k) | Maintain heap of size k |
| Merge k sorted lists | O(n log k) | O(k) | n = total elements |
| Find median (stream) | O(log n) per add | O(n) | Two heaps |

### 🧠 Interview Tips

- **Min vs max heap**: "Python has min heap by default, I'll negate for max heap"
- **K largest**: "I'll use min heap of size k to track k largest elements"
- **K smallest**: "I'll use max heap of size k to track k smallest elements"
- **Two heaps**: "For median, I'll maintain two heaps: max heap for smaller half, min heap for larger half"
- **Heap vs sorting**: "Heap is O(n log k) vs sorting O(n log n), better when k << n"
- **Custom comparison**: "I'll use tuples where first element is priority"

**Common follow-ups:**
- "Why heap instead of sorting?" → Better complexity when k << n
- "Can you do it with less space?" → Depends on problem, sometimes no
- "Handle stream of data?" → Heap is perfect for online algorithms

**Red flags to avoid:**
- Forgetting to negate for max heap in Python
- Not maintaining heap size in k largest/smallest
- Accessing empty heap
- Not handling ties in custom comparisons
- Confusing when to use min vs max heap

---

## 15. Greedy Algorithms

---

### ❓ When should I use this?

- Problem has **optimal substructure** and **greedy choice property**
- Keywords: "maximum", "minimum", "optimal", "interval", "scheduling"
- Making **locally optimal choice** leads to globally optimal solution
- Usually involves **sorting** and making greedy decisions
- **Proof required**: Must prove greedy choice is safe

**When greedy works:**
- Activity selection / interval scheduling
- Huffman coding
- Fractional knapsack
- Minimum spanning tree (Kruskal, Prim)

**When greedy fails:**
- 0/1 knapsack (need DP)
- Longest path in graph
- Coin change with arbitrary denominations

### 🧠 Core Idea (Intuition)

**Greedy**: Make the best choice at each step without looking back

**Key properties:**
1. **Greedy choice property**: Local optimum leads to global optimum
2. **Optimal substructure**: Optimal solution contains optimal subsolutions

**Mental model**: 
- Like climbing a hill by always going upward (may miss global peak)
- Like spending money: always use largest bill that fits (works for some currencies)

**Greedy vs DP:**
- Greedy: Make irrevocable choice at each step
- DP: Consider all choices, pick best

### 🧩 Common Problem Types

- Interval scheduling / meeting rooms
- Jump game
- Gas station
- Partition labels
- Non-overlapping intervals
- Minimum arrows to burst balloons
- Task scheduler
- Candy distribution
- Assign cookies
- Queue reconstruction

### 🧱 Template (Python)

```python
# Pattern 1: Interval Scheduling (Sort by End Time)
def interval_scheduling(intervals):
    # Sort by end time
    intervals.sort(key=lambda x: x[1])
    
    count = 0
    last_end = float('-inf')
    
    for start, end in intervals:
        if start >= last_end:
            count += 1
            last_end = end
    
    return count

# Pattern 2: Greedy with Sorting
def greedy_sort(items):
    # Sort by some criterion
    items.sort(key=lambda x: criterion(x))
    
    result = 0
    for item in items:
        if can_take(item):
            result += process(item)
    
    return result

# Pattern 3: Two Pointers Greedy
def two_pointer_greedy(arr1, arr2):
    arr1.sort()
    arr2.sort()
    
    i, j = 0, 0
    result = 0
    
    while i < len(arr1) and j < len(arr2):
        if condition(arr1[i], arr2[j]):
            result += 1
            i += 1
            j += 1
        else:
            j += 1  # or i += 1 depending on problem
    
    return result

# Pattern 4: Greedy with Priority Queue
def greedy_with_heap(items):
    import heapq
    
    items.sort(key=lambda x: x.start)
    heap = []
    
    for item in items:
        # Make greedy decision based on heap
        if heap and can_reuse(heap[0], item):
            heapq.heappop(heap)
        
        heapq.heappush(heap, item.end)
    
    return len(heap)

# Pattern 5: Greedy State Tracking
def greedy_state(arr):
    # Track current state
    current_state = initial_state
    result = 0
    
    for item in arr:
        # Make greedy choice
        if should_change_state(item, current_state):
            current_state = update_state(item)
            result += 1
    
    return result

# Pattern 6: Partition/Split Greedy
def partition_greedy(s):
    # Track last occurrence of each character
    last = {c: i for i, c in enumerate(s)}
    
    partitions = []
    start = 0
    end = 0
    
    for i, c in enumerate(s):
        end = max(end, last[c])
        
        if i == end:
            partitions.append(s[start:end+1])
            start = i + 1
    
    return partitions
```

### 📌 Step-by-Step Walkthrough

**Example 1: Jump Game [2,3,1,1,4]**

```
Can reach last index?

i=0, nums[0]=2, max_reach=2
  Can reach indices 1,2

i=1, nums[1]=3, max_reach=max(2, 1+3)=4
  Can reach indices 2,3,4 ✓

i=2, nums[2]=1, max_reach=max(4, 2+1)=4

Already can reach end (4 >= 4) → True
```

**Example 2: Non-overlapping Intervals [[1,2],[2,3],[3,4],[1,3]]**

```
Sort by end time: [[1,2],[2,3],[1,3],[3,4]]

last_end = -inf, remove_count = 0

[1,2]: 1 >= -inf ✓, last_end=2
[2,3]: 2 >= 2 ✓, last_end=3
[1,3]: 1 < 3 ✗, overlaps! remove_count=1
[3,4]: 3 >= 3 ✓, last_end=4

Remove count = 1
```

**Example 3: Partition Labels "ababcbacadefegdehijhklij"**

```
Last occurrence: {a:8, b:5, c:7, d:14, e:15, ...}

i=0, c='a', end=max(0,8)=8
i=1, c='b', end=max(8,5)=8
...
i=8, c='a', end=8, i==end → partition "ababcbaca"

i=9, c='d', end=max(9,14)=14
...
i=14, c='d', end=14, i==end → partition "defegde"

Continue...
Result: [9, 7, 8]
```

### 🧪 Solved Examples

**1. Jump Game**
```python
def canJump(nums):
    max_reach = 0
    
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
        if max_reach >= len(nums) - 1:
            return True
    
    return True
```

**2. Jump Game II (Minimum Jumps)**
```python
def jump(nums):
    jumps = 0
    current_end = 0
    farthest = 0
    
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        
        if i == current_end:
            jumps += 1
            current_end = farthest
    
    return jumps
```

**3. Gas Station**
```python
def canCompleteCircuit(gas, cost):
    if sum(gas) < sum(cost):
        return -1
    
    start = 0
    tank = 0
    
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        
        if tank < 0:
            start = i + 1
            tank = 0
    
    return start
```

**4. Partition Labels**
```python
def partitionLabels(s):
    last = {c: i for i, c in enumerate(s)}
    
    result = []
    start = 0
    end = 0
    
    for i, c in enumerate(s):
        end = max(end, last[c])
        
        if i == end:
            result.append(end - start + 1)
            start = i + 1
    
    return result
```

**5. Non-overlapping Intervals**
```python
def eraseOverlapIntervals(intervals):
    intervals.sort(key=lambda x: x[1])
    
    count = 0
    last_end = float('-inf')
    
    for start, end in intervals:
        if start >= last_end:
            last_end = end
        else:
            count += 1
    
    return count
```

**6. Minimum Number of Arrows to Burst Balloons**
```python
def findMinArrowShots(points):
    points.sort(key=lambda x: x[1])
    
    arrows = 0
    last_arrow = float('-inf')
    
    for start, end in points:
        if start > last_arrow:
            arrows += 1
            last_arrow = end
    
    return arrows
```

**7. Assign Cookies**
```python
def findContentChildren(g, s):
    g.sort()  # Greed factors
    s.sort()  # Cookie sizes
    
    child = 0
    cookie = 0
    
    while child < len(g) and cookie < len(s):
        if s[cookie] >= g[child]:
            child += 1
        cookie += 1
    
    return child
```

**8. Candy**
```python
def candy(ratings):
    n = len(ratings)
    candies = [1] * n
    
    # Left to right
    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1
    
    # Right to left
    for i in range(n-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candies[i] = max(candies[i], candies[i+1] + 1)
    
    return sum(candies)
```

**9. Queue Reconstruction by Height**
```python
def reconstructQueue(people):
    # Sort by height descending, then by k ascending
    people.sort(key=lambda x: (-x[0], x[1]))
    
    result = []
    for person in people:
        result.insert(person[1], person)
    
    return result
```

**10. Wiggle Subsequence**
```python
def wiggleMaxLength(nums):
    if len(nums) < 2:
        return len(nums)
    
    up = down = 1
    
    for i in range(1, len(nums)):
        if nums[i] > nums[i-1]:
            up = down + 1
        elif nums[i] < nums[i-1]:
            down = up + 1
    
    return max(up, down)
```

### ⚠️ Edge Cases & Pitfalls

- **Greedy doesn't always work**: Must prove correctness
- **Sorting changes order**: May lose original indices
- **Empty input**: Check `if not arr`
- **Single element**: May need special handling
- **Tie-breaking**: When sorting, secondary criteria matters
- **Off-by-one**: Careful with interval endpoints (inclusive/exclusive)
- **Negative numbers**: Some greedy strategies fail with negatives

### ⏱️ Time & Space Complexity

| Problem | Time | Space | Notes |
|---------|------|-------|-------|
| Jump game | O(n) | O(1) | Single pass |
| Gas station | O(n) | O(1) | Single pass |
| Interval scheduling | O(n log n) | O(1) | Sorting dominates |
| Partition labels | O(n) | O(1) | O(26) for last occurrence |
| Candy | O(n) | O(n) | Two passes |
| Assign cookies | O(n log n + m log m) | O(1) | Sorting both arrays |
| Queue reconstruction | O(n²) | O(n) | Insertions are O(n) |

### 🧠 Interview Tips

- **Prove greedy works**: "I'll sort by X because locally optimal choice leads to global optimum"
- **Explain sorting criterion**: "I sort by end time because it maximizes remaining time"
- **Compare to DP**: "Greedy works here because we don't need to reconsider choices"
- **Edge cases**: Empty, single element, all same
- **Complexity**: Mention sorting is usually O(n log n)

**Common follow-ups:**
- "Why does greedy work here?" → Explain greedy choice property
- "What if we can't sort?" → May need different approach
- "Can you prove correctness?" → Exchange argument or contradiction

**Red flags to avoid:**
- Assuming greedy works without justification
- Wrong sorting criterion
- Not handling edge cases
- Forgetting greedy fails for some problems (0/1 knapsack)

---

## 16.1. Graphs: BFS / DFS

---

### ❓ When should I use this?

- **Graph traversal**: Visit all vertices/edges
- **BFS**: Shortest path (unweighted), level-order, minimum steps
- **DFS**: Path finding, cycle detection, topological sort, connected components
- Keywords: "connected", "path", "reachable", "islands", "shortest"

**BFS vs DFS:**
- **BFS**: Queue, level by level, shortest path
- **DFS**: Stack/recursion, explore deep, less memory for sparse graphs

### 🧠 Core Idea (Intuition)

**Graph**: Vertices (nodes) connected by edges

**Representations:**
1. **Adjacency list**: `{node: [neighbors]}` - space efficient
2. **Adjacency matrix**: 2D array - O(1) edge lookup
3. **Edge list**: List of (u, v) pairs

**DFS**: Like exploring a maze by going as far as possible before backtracking

**BFS**: Like ripples in water, exploring layer by layer

**Mental model**:
- DFS: Depth-first, like reading a book chapter by chapter
- BFS: Breadth-first, like scanning each line of a page

### 🧩 Common Problem Types

**BFS:**
- Shortest path in unweighted graph
- Number of islands
- Word ladder
- Minimum knight moves
- Open the lock
- Walls and gates

**DFS:**
- Number of connected components
- Detect cycle
- Path exists
- Clone graph
- Course schedule (cycle detection)
- All paths from source to target

### 🧱 Template (Python)

```python
from collections import deque, defaultdict

# Pattern 1: BFS Template (Adjacency List)
def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        
        # Process node
        process(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Pattern 2: BFS with Distance/Level Tracking
def bfs_with_distance(graph, start):
    visited = {start: 0}  # node -> distance
    queue = deque([(start, 0)])
    
    while queue:
        node, dist = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    
    return visited

# Pattern 3: DFS Template (Recursive)
def dfs_recursive(graph, node, visited):
    if node in visited:
        return
    
    visited.add(node)
    
    # Process node
    process(node)
    
    for neighbor in graph[node]:
        dfs_recursive(graph, neighbor, visited)

# Pattern 4: DFS Template (Iterative)
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        
        if node in visited:
            continue
        
        visited.add(node)
        
        # Process node
        process(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)

# Pattern 5: Grid BFS (4-directional)
def grid_bfs(grid, start_r, start_c):
    rows, cols = len(grid), len(grid[0])
    visited = set([(start_r, start_c)])
    queue = deque([(start_r, start_c)])
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    
    while queue:
        r, c = queue.popleft()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and
                (nr, nc) not in visited and
                grid[nr][nc] == valid_cell):
                
                visited.add((nr, nc))
                queue.append((nr, nc))

# Pattern 6: Grid DFS (4-directional)
def grid_dfs(grid, r, c, visited):
    rows, cols = len(grid), len(grid[0])
    
    if (r < 0 or r >= rows or c < 0 or c >= cols or
        (r, c) in visited or grid[r][c] != valid_cell):
        return
    
    visited.add((r, c))
    
    # Process cell
    process(r, c)
    
    # Explore 4 directions
    grid_dfs(grid, r+1, c, visited)
    grid_dfs(grid, r-1, c, visited)
    grid_dfs(grid, r, c+1, visited)
    grid_dfs(grid, r, c-1, visited)

# Pattern 7: Multi-source BFS
def multi_source_bfs(grid):
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    visited = set()
    
    # Add all sources to queue
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == source:
                queue.append((r, c, 0))
                visited.add((r, c))
    
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    
    while queue:
        r, c, dist = queue.popleft()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and
                (nr, nc) not in visited):
                
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))

# Pattern 8: Bidirectional BFS (Shortest Path)
def bidirectional_bfs(graph, start, end):
    if start == end:
        return 0
    
    front = {start}
    back = {end}
    visited_front = {start}
    visited_back = {end}
    distance = 0
    
    while front and back:
        # Always expand smaller frontier
        if len(front) > len(back):
            front, back = back, front
            visited_front, visited_back = visited_back, visited_front
        
        distance += 1
        next_front = set()
        
        for node in front:
            for neighbor in graph[node]:
                if neighbor in back:
                    return distance
                
                if neighbor not in visited_front:
                    visited_front.add(neighbor)
                    next_front.add(neighbor)
        
        front = next_front
    
    return -1  # No path found
```

### 📌 Step-by-Step Walkthrough

**Example 1: BFS on Graph**

```
Graph: 0 -- 1 -- 2
       |    |
       3 -- 4

Start BFS from 0:

queue=[0], visited={0}
  Process 0, neighbors=[1,3]
  queue=[1,3], visited={0,1,3}

queue=[1,3], process 1, neighbors=[0,2,4]
  0 visited, add 2,4
  queue=[3,2,4], visited={0,1,2,3,4}

queue=[3,2,4], process 3, neighbors=[0,4]
  All visited
  queue=[2,4]

Continue until queue empty
Order: 0, 1, 3, 2, 4
```

**Example 2: Number of Islands (DFS)**

```
Grid:
1 1 0 0
1 0 0 1
0 0 1 1

DFS from (0,0):
  Mark (0,0), (0,1), (1,0) as visited
  Island 1 found

DFS from (1,3):
  Mark (1,3) as visited
  Island 2 found

DFS from (2,2):
  Mark (2,2), (2,3) as visited
  Island 3 found

Total: 3 islands
```

### 🧪 Solved Examples

**1. Number of Islands**
```python
def numIslands(grid):
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            grid[r][c] != '1'):
            return
        
        grid[r][c] = '0'  # Mark as visited
        
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                dfs(r, c)
    
    return islands
```

**2. Clone Graph**
```python
def cloneGraph(node):
    if not node:
        return None
    
    clones = {}
    
    def dfs(node):
        if node in clones:
            return clones[node]
        
        clone = Node(node.val)
        clones[node] = clone
        
        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)
```

**3. Course Schedule (Cycle Detection)**
```python
def canFinish(numCourses, prerequisites):
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    UNVISITED, VISITING, VISITED = 0, 1, 2
    state = [UNVISITED] * numCourses
    
    def has_cycle(node):
        if state[node] == VISITING:
            return True  # Cycle detected
        if state[node] == VISITED:
            return False
        
        state[node] = VISITING
        
        for neighbor in graph[node]:
            if has_cycle(neighbor):
                return True
        
        state[node] = VISITED
        return False
    
    for course in range(numCourses):
        if has_cycle(course):
            return False
    
    return True
```

**4. Word Ladder**
```python
def ladderLength(beginWord, endWord, wordList):
    wordSet = set(wordList)
    if endWord not in wordSet:
        return 0
    
    queue = deque([(beginWord, 1)])
    
    while queue:
        word, length = queue.popleft()
        
        if word == endWord:
            return length
        
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                
                if next_word in wordSet:
                    wordSet.remove(next_word)
                    queue.append((next_word, length + 1))
    
    return 0
```

**5. Pacific Atlantic Water Flow**
```python
def pacificAtlantic(heights):
    if not heights:
        return []
    
    rows, cols = len(heights), len(heights[0])
    
    def dfs(r, c, reachable):
        reachable.add((r, c))
        
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and
                (nr, nc) not in reachable and
                heights[nr][nc] >= heights[r][c]):
                
                dfs(nr, nc, reachable)
    
    pacific = set()
    atlantic = set()
    
    for c in range(cols):
        dfs(0, c, pacific)
        dfs(rows-1, c, atlantic)
    
    for r in range(rows):
        dfs(r, 0, pacific)
        dfs(r, cols-1, atlantic)
    
    return list(pacific & atlantic)
```

**6. Surrounded Regions**
```python
def solve(board):
    if not board:
        return
    
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] != 'O'):
            return
        
        board[r][c] = 'T'  # Temporary mark
        
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
    
    # Mark border-connected 'O's
    for r in range(rows):
        dfs(r, 0)
        dfs(r, cols-1)
    
    for c in range(cols):
        dfs(0, c)
        dfs(rows-1, c)
    
    # Flip surrounded 'O's to 'X', restore 'T's to 'O'
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'O':
                board[r][c] = 'X'
            elif board[r][c] == 'T':
                board[r][c] = 'O'
```

**7. All Paths From Source to Target**
```python
def allPathsSourceTarget(graph):
    n = len(graph)
    paths = []
    
    def dfs(node, path):
        if node == n - 1:
            paths.append(path[:])
            return
        
        for neighbor in graph[node]:
            path.append(neighbor)
            dfs(neighbor, path)
            path.pop()
    
    dfs(0, [0])
    return paths
```

**8. Shortest Bridge**
```python
def shortestBridge(grid):
    n = len(grid)
    
    def dfs(r, c, queue):
        if (r < 0 or r >= n or c < 0 or c >= n or
            grid[r][c] != 1):
            return
        
        grid[r][c] = 2
        queue.append((r, c, 0))
        
        dfs(r+1, c, queue)
        dfs(r-1, c, queue)
        dfs(r, c+1, queue)
        dfs(r, c-1, queue)
    
    # Find first island with DFS
    queue = deque()
    found = False
    for r in range(n):
        if found:
            break
        for c in range(n):
            if grid[r][c] == 1:
                dfs(r, c, queue)
                found = True
                break
    
    # BFS to find second island
    while queue:
        r, c, dist = queue.popleft()
        
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < n and 0 <= nc < n:
                if grid[nr][nc] == 1:
                    return dist
                if grid[nr][nc] == 0:
                    grid[nr][nc] = 2
                    queue.append((nr, nc, dist + 1))
    
    return -1
```

### ⚠️ Edge Cases & Pitfalls

- **Empty graph**: Check `if not graph`
- **Disconnected components**: May need to run BFS/DFS from multiple sources
- **Cycles**: DFS needs visited set to avoid infinite loops
- **Grid boundaries**: Always check bounds
- **Visited tracking**: Mark as visited when adding to queue (BFS), not when processing
- **Modifying input**: Clarify if you can modify grid in-place
- **Stack overflow**: DFS recursion may overflow for large graphs (use iterative)

### ⏱️ Time & Space Complexity

| Algorithm | Time | Space | Notes |
|-----------|------|-------|-------|
| BFS | O(V + E) | O(V) | V=vertices, E=edges |
| DFS | O(V + E) | O(V) | Recursion stack or explicit stack |
| Grid BFS | O(m×n) | O(m×n) | m×n = grid size |
| Grid DFS | O(m×n) | O(m×n) | Worst case: all cells visited |
| Multi-source BFS | O(m×n) | O(m×n) | All cells may be in queue |

### 🧠 Interview Tips

- **BFS for shortest path**: "BFS guarantees shortest path in unweighted graph"
- **DFS for connectivity**: "DFS is good for finding connected components"
- **Visited set**: "I'll mark nodes as visited to avoid cycles"
- **Grid as graph**: "I'll treat grid as graph with 4-directional edges"
- **Multi-source**: "I'll add all sources to queue initially"
- **Space optimization**: "For grid, I can modify in-place to save space"

**Common follow-ups:**
- "What if graph has cycles?" → Need visited set
- "Shortest path?" → Use BFS
- "Can you do it iteratively?" → Show iterative DFS with stack
- "What if grid is very large?" → Discuss memory constraints

**Red flags to avoid:**
- Not handling cycles in DFS
- Marking visited too late in BFS (duplicates in queue)
- Forgetting boundary checks in grid
- Stack overflow in recursive DFS
- Not considering disconnected components

---

## 16.2. Graphs: Topological Sort

---

### ❓ When should I use this?

- **Directed Acyclic Graph (DAG)** with dependencies
- Keywords: "ordering", "prerequisites", "dependencies", "schedule", "course order"
- Need to find **linear ordering** where all edges point forward
- **Cycle detection**: If cycle exists, no topological order possible

**Applications:**
- Course schedule (prerequisites)
- Build systems (compilation order)
- Task scheduling with dependencies

### 🧠 Core Idea (Intuition)

**Topological sort**: Linear ordering of vertices such that for every edge (u, v), u comes before v

**Two algorithms:**
1. **Kahn's Algorithm (BFS)**: Process nodes with in-degree 0
2. **DFS-based**: Post-order DFS, reverse the result

**Mental model**: 
- Like getting dressed: underwear before pants, socks before shoes
- Like cooking: prep ingredients before cooking

### 🧩 Common Problem Types

- Course schedule I & II
- Alien dictionary
- Sequence reconstruction
- Minimum height trees
- Build order (tasks with dependencies)
- Parallel courses

### 🧱 Template (Python)

```python
from collections import deque, defaultdict

# Pattern 1: Kahn's Algorithm (BFS-based)
def topological_sort_kahn(n, edges):
    # Build graph and in-degree
    graph = defaultdict(list)
    in_degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    # Start with nodes having in-degree 0
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        # Reduce in-degree of neighbors
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # If result doesn't contain all nodes, there's a cycle
    return result if len(result) == n else []

# Pattern 2: DFS-based Topological Sort
def topological_sort_dfs(n, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
    
    UNVISITED, VISITING, VISITED = 0, 1, 2
    state = [UNVISITED] * n
    result = []
    
    def dfs(node):
        if state[node] == VISITING:
            return False  # Cycle detected
        if state[node] == VISITED:
            return True
        
        state[node] = VISITING
        
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        
        state[node] = VISITED
        result.append(node)  # Add in post-order
        return True
    
    for i in range(n):
        if state[i] == UNVISITED:
            if not dfs(i):
                return []  # Cycle detected
    
    return result[::-1]  # Reverse for topological order

# Pattern 3: Course Schedule (Cycle Detection)
def can_finish(numCourses, prerequisites):
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    count = 0
    
    while queue:
        node = queue.popleft()
        count += 1
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return count == numCourses

# Pattern 4: All Topological Orders (Backtracking)
def all_topological_sorts(n, edges):
    graph = defaultdict(list)
    in_degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    result = []
    
    def backtrack(path, in_deg):
        if len(path) == n:
            result.append(path[:])
            return
        
        for node in range(n):
            if in_deg[node] == 0:
                # Choose node
                path.append(node)
                new_in_deg = in_deg[:]
                new_in_deg[node] = -1  # Mark as used
                
                for neighbor in graph[node]:
                    new_in_deg[neighbor] -= 1
                
                backtrack(path, new_in_deg)
                
                # Unchoose
                path.pop()
    
    backtrack([], in_degree)
    return result
```

### 📌 Step-by-Step Walkthrough

**Example: Course Schedule II with prerequisites [[1,0],[2,0],[3,1],[3,2]]**

```
Courses: 0, 1, 2, 3
Edges: 0→1, 0→2, 1→3, 2→3

Build graph and in-degree:
  graph: {0: [1,2], 1: [3], 2: [3]}
  in_degree: [0, 1, 1, 2]

Kahn's Algorithm:

Initial: queue=[0] (in-degree 0)
  result=[]

Step 1: Process 0
  result=[0]
  Reduce in-degree: 1→0, 2→0
  queue=[1,2]

Step 2: Process 1
  result=[0,1]
  Reduce in-degree: 3→1
  queue=[2]

Step 3: Process 2
  result=[0,1,2]
  Reduce in-degree: 3→0
  queue=[3]

Step 4: Process 3
  result=[0,1,2,3]
  queue=[]

All courses processed → Valid order: [0,1,2,3]
```

**Example: Cycle Detection**

```
Graph: 0→1→2→0 (cycle)

DFS from 0:
  state[0] = VISITING
  Visit 1:
    state[1] = VISITING
    Visit 2:
      state[2] = VISITING
      Visit 0:
        state[0] = VISITING ✗ (cycle!)

Cycle detected → No topological order
```

### 🧪 Solved Examples

**1. Course Schedule**
```python
def canFinish(numCourses, prerequisites):
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    count = 0
    
    while queue:
        course = queue.popleft()
        count += 1
        
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return count == numCourses
```

**2. Course Schedule II**
```python
def findOrder(numCourses, prerequisites):
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    result = []
    
    while queue:
        course = queue.popleft()
        result.append(course)
        
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return result if len(result) == numCourses else []
```

**3. Alien Dictionary**
```python
def alienOrder(words):
    # Build graph from word order
    graph = defaultdict(set)
    in_degree = {c: 0 for word in words for c in word}
    
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        min_len = min(len(w1), len(w2))
        
        # Check invalid case: prefix comes after
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""
        
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in graph[w1[j]]:
                    graph[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                break
    
    # Topological sort
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []
    
    while queue:
        c = queue.popleft()
        result.append(c)
        
        for neighbor in graph[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return ''.join(result) if len(result) == len(in_degree) else ""
```

**4. Sequence Reconstruction**
```python
def sequenceReconstruction(nums, sequences):
    graph = defaultdict(list)
    in_degree = {num: 0 for num in nums}
    
    for seq in sequences:
        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i + 1]
            if u not in in_degree or v not in in_degree:
                return False
            graph[u].append(v)
            in_degree[v] += 1
    
    queue = deque([num for num in nums if in_degree[num] == 0])
    result = []
    
    while queue:
        if len(queue) > 1:
            return False  # Not unique
        
        num = queue.popleft()
        result.append(num)
        
        for neighbor in graph[num]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result == nums
```

**5. Parallel Courses**
```python
def minimumSemesters(n, relations):
    graph = defaultdict(list)
    in_degree = [0] * (n + 1)
    
    for prev, next in relations:
        graph[prev].append(next)
        in_degree[next] += 1
    
    queue = deque([i for i in range(1, n + 1) if in_degree[i] == 0])
    semesters = 0
    studied = 0
    
    while queue:
        semesters += 1
        for _ in range(len(queue)):
            course = queue.popleft()
            studied += 1
            
            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
    
    return semesters if studied == n else -1
```

### ⚠️ Edge Cases & Pitfalls

- **Cycle detection**: No topological order if cycle exists
- **Multiple valid orders**: Topological sort is not unique
- **Disconnected components**: May have multiple starting nodes
- **Empty graph**: Handle n=0 or no edges
- **Self-loops**: Create cycles
- **Invalid input**: Nodes not in range [0, n-1]

### ⏱️ Time & Space Complexity

| Algorithm | Time | Space | Notes |
|-----------|------|-------|-------|
| Kahn's (BFS) | O(V + E) | O(V + E) | V=vertices, E=edges |
| DFS-based | O(V + E) | O(V + E) | Recursion stack |
| Cycle detection | O(V + E) | O(V + E) | Part of topological sort |
| All orders | O(V! × V) | O(V) | Exponential (backtracking) |

### 🧠 Interview Tips

- **Choose algorithm**: "I'll use Kahn's algorithm (BFS) for clearer cycle detection"
- **Explain in-degree**: "In-degree tracks number of prerequisites"
- **Cycle detection**: "If we can't process all nodes, there's a cycle"
- **Multiple orders**: "Topological sort is not unique if multiple nodes have in-degree 0"
- **DFS alternative**: "I could also use DFS with three states: unvisited, visiting, visited"

**Common follow-ups:**
- "What if there's a cycle?" → Return empty or indicate error
- "Can there be multiple valid orders?" → Yes, explain when
- "How do you detect cycles?" → Check if all nodes processed
- "DFS vs BFS?" → Both work, BFS more intuitive for this problem

**Red flags to avoid:**
- Not checking if all nodes processed (missing cycle detection)
- Confusing in-degree with out-degree
- Not handling disconnected components
- Forgetting to reverse DFS result

---

## 16.3. Graphs: Union-Find (Disjoint Set Union)

---

### ❓ When should I use this?

- Need to track **connected components** dynamically
- Keywords: "connected", "union", "find", "group", "redundant connection"
- **Dynamic connectivity**: Elements join groups over time
- Efficiently answer "are X and Y in same group?"
- **Cycle detection** in undirected graphs

### 🧠 Core Idea (Intuition)

**Union-Find**: Data structure that tracks disjoint sets

**Two operations:**
1. **Find**: Which set does element belong to? (find root)
2. **Union**: Merge two sets (connect roots)

**Optimizations:**
1. **Path compression**: Make tree flat during find
2. **Union by rank/size**: Attach smaller tree to larger

**Mental model**: 
- Like social groups where each group has a leader
- Like family trees where you find the ancestor

### 🧩 Common Problem Types

- Number of connected components
- Redundant connection (cycle in graph)
- Accounts merge
- Smallest string with swaps
- Number of provinces
- Graph valid tree
- Most stones removed
- Satisfiability of equality equations

### 🧱 Template (Python)

```python
# Pattern 1: Basic Union-Find
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # Number of components
    
    def find(self, x):
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.count -= 1
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
    
    def get_count(self):
        return self.count

# Pattern 2: Union-Find with Size
class UnionFindWithSize:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by size
        if self.size[root_x] < self.size[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        
        return True
    
    def get_size(self, x):
        return self.size[self.find(x)]

# Pattern 3: Union-Find for Grid
class UnionFindGrid:
    def __init__(self, grid):
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.parent = {}
        self.rank = {}
        
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] == 1:
                    cell = (r, c)
                    self.parent[cell] = cell
                    self.rank[cell] = 0
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True

# Pattern 4: Weighted Union-Find (for equations)
class WeightedUnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.weight = [0] * n  # Weight to parent
    
    def find(self, x):
        if self.parent[x] != x:
            original_parent = self.parent[x]
            self.parent[x] = self.find(original_parent)
            self.weight[x] += self.weight[original_parent]
        return self.parent[x]
    
    def union(self, x, y, w):
        # x = y + w
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        self.parent[root_x] = root_y
        self.weight[root_x] = self.weight[y] - self.weight[x] + w
    
    def diff(self, x, y):
        if self.find(x) != self.find(y):
            return None
        return self.weight[x] - self.weight[y]
```

### 📌 Step-by-Step Walkthrough

**Example: Union-Find Operations**

```
Initialize with n=5: [0,1,2,3,4]
  parent: [0,1,2,3,4]
  rank:   [0,0,0,0,0]

Union(0,1):
  find(0)=0, find(1)=1
  Same rank, attach 1 to 0
  parent: [0,0,2,3,4]
  rank:   [1,0,0,0,0]

Union(2,3):
  find(2)=2, find(3)=3
  Same rank, attach 3 to 2
  parent: [0,0,2,2,4]
  rank:   [1,0,1,0,0]

Union(0,2):
  find(0)=0, find(2)=2
  Same rank, attach 2 to 0
  parent: [0,0,0,2,4]
  rank:   [2,0,1,0,0]

Connected(1,3)?
  find(1)=0, find(3)=2→0
  Both have root 0 → Yes!

Tree structure:
     0
    / \
   1   2
       |
       3
```

**Example: Redundant Connection [[1,2],[1,3],[2,3]]**

```
Edges: (1,2), (1,3), (2,3)

Process (1,2):
  union(1,2) → Success
  Components: {1,2}, {3}

Process (1,3):
  union(1,3) → Success
  Components: {1,2,3}

Process (2,3):
  find(2)=1, find(3)=1
  Already connected! → This is redundant edge

Answer: [2,3]
```

### 🧪 Solved Examples

**1. Number of Connected Components**
```python
def countComponents(n, edges):
    uf = UnionFind(n)
    
    for u, v in edges:
        uf.union(u, v)
    
    return uf.get_count()
```

**2. Redundant Connection**
```python
def findRedundantConnection(edges):
    uf = UnionFind(len(edges) + 1)
    
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]
    
    return []
```

**3. Graph Valid Tree**
```python
def validTree(n, edges):
    if len(edges) != n - 1:
        return False
    
    uf = UnionFind(n)
    
    for u, v in edges:
        if not uf.union(u, v):
            return False  # Cycle detected
    
    return uf.get_count() == 1
```

**4. Number of Provinces**
```python
def findCircleNum(isConnected):
    n = len(isConnected)
    uf = UnionFind(n)
    
    for i in range(n):
        for j in range(i + 1, n):
            if isConnected[i][j]:
                uf.union(i, j)
    
    return uf.get_count()
```

**5. Accounts Merge**
```python
def accountsMerge(accounts):
    uf = UnionFind(len(accounts))
    email_to_id = {}
    
    # Build union-find
    for i, account in enumerate(accounts):
        for email in account[1:]:
            if email in email_to_id:
                uf.union(i, email_to_id[email])
            else:
                email_to_id[email] = i
    
    # Group emails by root
    groups = defaultdict(list)
    for email, idx in email_to_id.items():
        root = uf.find(idx)
        groups[root].append(email)
    
    # Build result
    result = []
    for idx, emails in groups.items():
        result.append([accounts[idx][0]] + sorted(emails))
    
    return result
```

**6. Most Stones Removed**
```python
def removeStones(stones):
    uf = UnionFind(20001)  # Max coordinate value
    
    for x, y in stones:
        uf.union(x, y + 10000)  # Offset y to avoid collision
    
    # Count unique roots
    roots = set()
    for x, y in stones:
        roots.add(uf.find(x))
    
    return len(stones) - len(roots)
```

**7. Satisfiability of Equality Equations**
```python
def equationsPossible(equations):
    uf = UnionFind(26)
    
    # Process equality first
    for eq in equations:
        if eq[1] == '=':
            x = ord(eq[0]) - ord('a')
            y = ord(eq[3]) - ord('a')
            uf.union(x, y)
    
    # Check inequality
    for eq in equations:
        if eq[1] == '!':
            x = ord(eq[0]) - ord('a')
            y = ord(eq[3]) - ord('a')
            if uf.connected(x, y):
                return False
    
    return True
```

**8. Smallest String With Swaps**
```python
def smallestStringWithSwaps(s, pairs):
    n = len(s)
    uf = UnionFind(n)
    
    # Union indices that can be swapped
    for i, j in pairs:
        uf.union(i, j)
    
    # Group indices by root
    groups = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        groups[root].append(i)
    
    # Sort characters in each group
    result = list(s)
    for indices in groups.values():
        chars = sorted([s[i] for i in indices])
        indices.sort()
        for i, char in zip(indices, chars):
            result[i] = char
    
    return ''.join(result)
```

**9. Evaluate Division**
```python
def calcEquation(equations, values, queries):
    uf = {}
    
    def find(x):
        if x not in uf:
            uf[x] = (x, 1.0)
        if uf[x][0] != x:
            root, weight = find(uf[x][0])
            uf[x] = (root, weight * uf[x][1])
        return uf[x]
    
    def union(x, y, value):
        root_x, weight_x = find(x)
        root_y, weight_y = find(y)
        
        if root_x != root_y:
            uf[root_x] = (root_y, value * weight_y / weight_x)
    
    # Build union-find
    for (x, y), value in zip(equations, values):
        union(x, y, value)
    
    # Answer queries
    result = []
    for x, y in queries:
        if x not in uf or y not in uf:
            result.append(-1.0)
        else:
            root_x, weight_x = find(x)
            root_y, weight_y = find(y)
            if root_x != root_y:
                result.append(-1.0)
            else:
                result.append(weight_x / weight_y)
    
    return result
```

### ⚠️ Edge Cases & Pitfalls

- **Self-loops**: Union(x, x) should be handled
- **Invalid indices**: Check bounds
- **Already connected**: Union returns false if already in same set
- **Path compression**: Essential for O(α(n)) complexity
- **Union by rank vs size**: Both work, choose one consistently
- **Disconnected components**: May have multiple roots
- **Grid problems**: Convert 2D coordinates to 1D or use tuples

### ⏱️ Time & Space Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Find (with compression) | O(α(n)) | O(1) | α(n) ≈ inverse Ackermann |
| Union (with rank) | O(α(n)) | O(1) | Amortized |
| Build UF | O(n) | O(n) | Initialize n elements |
| Process m edges | O(m × α(n)) | O(n) | m union operations |

α(n) is effectively constant (< 5) for all practical values of n.

### 🧠 Interview Tips

- **Explain optimizations**: "I'll use path compression and union by rank for O(α(n)) operations"
- **When to use**: "Union-Find is perfect for dynamic connectivity queries"
- **Cycle detection**: "If union returns false, we've found a cycle"
- **Count components**: "Track count, decrement on successful union"
- **Alternative**: "Could also use DFS, but Union-Find is more efficient for dynamic updates"

**Common follow-ups:**
- "What's the time complexity?" → O(α(n)) per operation
- "Why not DFS?" → Union-Find handles dynamic updates better
- "What if graph changes?" → Union-Find supports incremental updates
- "Can you disconnect nodes?" → Standard Union-Find doesn't support this

**Red flags to avoid:**
- Not implementing path compression (O(n) per find)
- Not using union by rank/size (unbalanced trees)
- Forgetting to check if union succeeds (cycle detection)
- Incorrect parent initialization

---

## 16.4. Graphs: Shortest Path (Dijkstra, Bellman-Ford)

---

### ❓ When should I use this?

- Find **shortest path** in **weighted graph**
- **Dijkstra**: Non-negative weights, O((V+E) log V)
- **Bellman-Ford**: Handles negative weights, detects negative cycles, O(VE)
- **Floyd-Warshall**: All-pairs shortest path, O(V³)

**Choose algorithm:**
- Non-negative weights → Dijkstra (faster)
- Negative weights → Bellman-Ford
- All pairs → Floyd-Warshall
- Unweighted → BFS

### 🧠 Core Idea (Intuition)

**Dijkstra**: Greedy algorithm using priority queue
- Always expand closest unvisited node
- Like spreading ink from source, always expanding nearest frontier

**Bellman-Ford**: Dynamic programming
- Relax all edges V-1 times
- Like ripples in water, distance improves with each iteration

**Mental model**:
- Dijkstra: Like GPS finding fastest route, always taking nearest unexplored road
- Bellman-Ford: Like trying all possible routes, keeping track of shortest

### 🧩 Common Problem Types

- Network delay time
- Cheapest flights within K stops
- Path with minimum effort
- Swim in rising water
- Minimum cost to reach destination
- Find the city with smallest number of neighbors
- Reconstruct itinerary

### 🧱 Template (Python)

```python
import heapq
from collections import defaultdict

# Pattern 1: Dijkstra's Algorithm (Priority Queue)
def dijkstra(graph, start, n):
    # graph[u] = [(v, weight), ...]
    dist = [float('inf')] * n
    dist[start] = 0
    
    heap = [(0, start)]  # (distance, node)
    
    while heap:
        d, u = heapq.heappop(heap)
        
        if d > dist[u]:
            continue  # Already found better path
        
        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    
    return dist

# Pattern 2: Dijkstra with Path Reconstruction
def dijkstra_with_path(graph, start, end, n):
    dist = [float('inf')] * n
    dist[start] = 0
    parent = [-1] * n
    
    heap = [(0, start)]
    
    while heap:
        d, u = heapq.heappop(heap)
        
        if u == end:
            break
        
        if d > dist[u]:
            continue
        
        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            
            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(heap, (new_dist, v))
    
    # Reconstruct path
    path = []
    current = end
    while current != -1:
        path.append(current)
        current = parent[current]
    
    return dist[end], path[::-1]

# Pattern 3: Bellman-Ford Algorithm
def bellman_ford(edges, start, n):
    # edges = [(u, v, weight), ...]
    dist = [float('inf')] * n
    dist[start] = 0
    
    # Relax edges V-1 times
    for _ in range(n - 1):
        for u, v, weight in edges:
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
    
    # Check for negative cycle
    for u, v, weight in edges:
        if dist[u] != float('inf') and dist[u] + weight < dist[v]:
            return None  # Negative cycle detected
    
    return dist

# Pattern 4: Dijkstra on Grid (2D)
def dijkstra_grid(grid):
    rows, cols = len(grid), len(grid[0])
    dist = [[float('inf')] * cols for _ in range(rows)]
    dist[0][0] = grid[0][0]
    
    heap = [(grid[0][0], 0, 0)]  # (cost, row, col)
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    
    while heap:
        d, r, c = heapq.heappop(heap)
        
        if d > dist[r][c]:
            continue
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols:
                new_dist = dist[r][c] + grid[nr][nc]
                
                if new_dist < dist[nr][nc]:
                    dist[nr][nc] = new_dist
                    heapq.heappush(heap, (new_dist, nr, nc))
    
    return dist[rows-1][cols-1]

# Pattern 5: Modified Dijkstra (K Stops)
def dijkstra_k_stops(graph, start, end, k, n):
    # (cost, node, stops)
    heap = [(0, start, 0)]
    # dist[node][stops] = min cost with at most 'stops' stops
    dist = [[float('inf')] * (k + 2) for _ in range(n)]
    dist[start][0] = 0
    
    while heap:
        cost, u, stops = heapq.heappop(heap)
        
        if u == end:
            return cost
        
        if stops > k:
            continue
        
        for v, price in graph[u]:
            new_cost = cost + price
            
            if new_cost < dist[v][stops + 1]:
                dist[v][stops + 1] = new_cost
                heapq.heappush(heap, (new_cost, v, stops + 1))
    
    return -1

# Pattern 6: Floyd-Warshall (All Pairs)
def floyd_warshall(graph, n):
    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for u, v, weight in graph:
        dist[u][v] = weight
    
    # Try all intermediate vertices
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

# Pattern 7: A* Algorithm (Heuristic-based)
def a_star(graph, start, end, heuristic):
    # heuristic(node) estimates distance to end
    g_score = {start: 0}  # Cost from start
    f_score = {start: heuristic(start)}  # g + h
    
    heap = [(f_score[start], start)]
    came_from = {}
    
    while heap:
        _, current = heapq.heappop(heap)
        
        if current == end:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for neighbor, cost in graph[current]:
            tentative_g = g_score[current] + cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor)
                heapq.heappush(heap, (f_score[neighbor], neighbor))
    
    return None
```

### 📌 Step-by-Step Walkthrough

**Example: Dijkstra on Graph**

```
Graph: 0--1--2
       |  |  |
       3--4--5

Edges with weights:
0-1: 4, 0-3: 2
1-2: 3, 1-4: 2
2-5: 1
3-4: 3
4-5: 2

Start from 0:

Initial: dist=[0,∞,∞,∞,∞,∞], heap=[(0,0)]

Step 1: Process 0 (dist=0)
  Update 1: dist[1]=4, heap=[(2,3),(4,1)]
  Update 3: dist[3]=2

Step 2: Process 3 (dist=2)
  Update 4: dist[4]=5, heap=[(4,1),(5,4)]

Step 3: Process 1 (dist=4)
  Update 2: dist[2]=7, heap=[(5,4),(6,2),(7,2)]
  Update 4: dist[4]=6→5 (no update)

Step 4: Process 4 (dist=5)
  Update 5: dist[5]=7, heap=[(6,2),(7,2),(7,5)]

Continue...

Final: dist=[0,4,7,2,5,7]
```

### 🧪 Solved Examples

**1. Network Delay Time**
```python
def networkDelayTime(times, n, k):
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    
    dist = [float('inf')] * (n + 1)
    dist[k] = 0
    heap = [(0, k)]
    
    while heap:
        d, u = heapq.heappop(heap)
        
        if d > dist[u]:
            continue
        
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    
    max_dist = max(dist[1:])
    return max_dist if max_dist < float('inf') else -1
```

**2. Cheapest Flights Within K Stops**
```python
def findCheapestPrice(n, flights, src, dst, k):
    graph = defaultdict(list)
    for u, v, price in flights:
        graph[u].append((v, price))
    
    heap = [(0, src, 0)]  # (cost, node, stops)
    dist = [[float('inf')] * (k + 2) for _ in range(n)]
    dist[src][0] = 0
    
    while heap:
        cost, u, stops = heapq.heappop(heap)
        
        if u == dst:
            return cost
        
        if stops > k:
            continue
        
        for v, price in graph[u]:
            new_cost = cost + price
            if new_cost < dist[v][stops + 1]:
                dist[v][stops + 1] = new_cost
                heapq.heappush(heap, (new_cost, v, stops + 1))
    
    return -1
```

**3. Path With Minimum Effort**
```python
def minimumEffortPath(heights):
    rows, cols = len(heights), len(heights[0])
    effort = [[float('inf')] * cols for _ in range(rows)]
    effort[0][0] = 0
    
    heap = [(0, 0, 0)]  # (effort, row, col)
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    
    while heap:
        e, r, c = heapq.heappop(heap)
        
        if r == rows - 1 and c == cols - 1:
            return e
        
        if e > effort[r][c]:
            continue
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols:
                new_effort = max(e, abs(heights[nr][nc] - heights[r][c]))
                
                if new_effort < effort[nr][nc]:
                    effort[nr][nc] = new_effort
                    heapq.heappush(heap, (new_effort, nr, nc))
    
    return 0
```

**4. Swim in Rising Water**
```python
def swimInWater(grid):
    n = len(grid)
    visited = set([(0, 0)])
    heap = [(grid[0][0], 0, 0)]
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    
    while heap:
        time, r, c = heapq.heappop(heap)
        
        if r == n - 1 and c == n - 1:
            return time
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < n and 0 <= nc < n and
                (nr, nc) not in visited):
                
                visited.add((nr, nc))
                heapq.heappush(heap, (max(time, grid[nr][nc]), nr, nc))
    
    return -1
```

**5. Find the City With Smallest Number of Neighbors**
```python
def findTheCity(n, edges, distanceThreshold):
    # Floyd-Warshall
    dist = [[float('inf')] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for u, v, w in edges:
        dist[u][v] = w
        dist[v][u] = w
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # Count reachable cities
    min_count = n
    result = 0
    
    for i in range(n):
        count = sum(1 for j in range(n) if i != j and dist[i][j] <= distanceThreshold)
        if count <= min_count:
            min_count = count
            result = i
    
    return result
```

### ⚠️ Edge Cases & Pitfalls

- **Negative weights**: Dijkstra doesn't work, use Bellman-Ford
- **Negative cycles**: Bellman-Ford can detect
- **Disconnected graph**: Some nodes unreachable (dist = ∞)
- **Self-loops**: Usually ignored
- **Multiple edges**: Keep minimum weight
- **Priority queue duplicates**: Check if distance improved
- **Integer overflow**: Large weights may overflow

### ⏱️ Time & Space Complexity

| Algorithm | Time | Space | Notes |
|-----------|------|-------|-------|
| Dijkstra | O((V+E) log V) | O(V) | With min-heap |
| Bellman-Ford | O(VE) | O(V) | Slower but handles negatives |
| Floyd-Warshall | O(V³) | O(V²) | All-pairs shortest path |
| A* | O((V+E) log V) | O(V) | Better with good heuristic |

### 🧠 Interview Tips

- **Choose algorithm**: "Non-negative weights, so I'll use Dijkstra with min-heap"
- **Explain priority queue**: "Always process closest unvisited node"
- **Path reconstruction**: "I'll track parent pointers to rebuild path"
- **Negative weights**: "If there are negative weights, I'd use Bellman-Ford"
- **Grid as graph**: "I'll treat grid cells as nodes with 4-directional edges"

**Common follow-ups:**
- "What if weights are negative?" → Use Bellman-Ford
- "All-pairs shortest path?" → Floyd-Warshall
- "Can you reconstruct the path?" → Track parent pointers
- "What if graph is very large?" → Discuss memory constraints, A* with heuristic

**Red flags to avoid:**
- Using Dijkstra with negative weights
- Not checking if distance improved (duplicate processing)
- Forgetting to handle unreachable nodes
- Integer overflow with large weights

---

## 17. Dynamic Programming

---

### ❓ When should I use this?

- Problem has **optimal substructure** and **overlapping subproblems**
- Keywords: "maximum", "minimum", "count ways", "longest", "shortest"
- **Optimization problems**: Find best solution
- **Counting problems**: Count number of ways
- **Decision problems**: Is it possible?
- Can be solved by **breaking into subproblems** that are reused

**DP vs Greedy vs Backtracking:**
- **DP**: Overlapping subproblems, optimal substructure
- **Greedy**: Local optimum leads to global (no backtracking needed)
- **Backtracking**: Explore all possibilities (no overlapping subproblems)

### 🧠 Core Idea (Intuition)

**Dynamic Programming**: Solve problem by combining solutions to subproblems

**Two approaches:**
1. **Top-down (Memoization)**: Recursion + cache
2. **Bottom-up (Tabulation)**: Iterative, fill table

**Steps to solve DP:**
1. Define state (what does dp[i] represent?)
2. Find recurrence relation (how to compute dp[i] from previous states?)
3. Identify base cases
4. Determine order of computation
5. Optimize space if possible

**Mental model**: Like climbing stairs, each step depends on previous steps

---

## 17.1. 1D DP

---

### ❓ When should I use this?

- State depends on **single variable** (usually index or value)
- Keywords: "sequence", "array", "string", "climb", "rob"
- `dp[i]` represents optimal solution up to index i

### 🧩 Common Problem Types

- Climbing stairs
- House robber
- Decode ways
- Coin change
- Jump game
- Longest increasing subsequence
- Word break
- Partition equal subset sum

### 🧱 Template (Python)

```python
# Pattern 1: Basic 1D DP (Bottom-up)
def dp_1d_basic(arr):
    n = len(arr)
    dp = [0] * (n + 1)
    
    # Base case
    dp[0] = base_value
    
    # Fill dp table
    for i in range(1, n + 1):
        dp[i] = compute_from_previous(dp, i)
    
    return dp[n]

# Pattern 2: 1D DP with Top-down (Memoization)
def dp_1d_memoization(arr):
    memo = {}
    
    def helper(i):
        if i in memo:
            return memo[i]
        
        if i == base_case:
            return base_value
        
        memo[i] = compute_from_previous(helper, i)
        return memo[i]
    
    return helper(len(arr))

# Pattern 3: Space-Optimized 1D DP (Keep Last k States)
def dp_1d_optimized(arr):
    n = len(arr)
    
    # Only keep last 2 states (for Fibonacci-like problems)
    prev2 = base_value_1
    prev1 = base_value_2
    
    for i in range(2, n + 1):
        current = compute_from(prev1, prev2)
        prev2 = prev1
        prev1 = current
    
    return prev1

# Pattern 4: 1D DP with Multiple Choices
def dp_1d_choices(arr):
    n = len(arr)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        # Try all possible choices
        for choice in get_choices(i):
            dp[i] = min(dp[i], dp[i - choice] + cost)
    
    return dp[n]

# Pattern 5: 1D DP with State Machine
def dp_state_machine(arr):
    n = len(arr)
    # Multiple states at each position
    state1 = [0] * n
    state2 = [0] * n
    
    # Base case
    state1[0] = initial_state1
    state2[0] = initial_state2
    
    for i in range(1, n):
        state1[i] = max(state1[i-1] + arr[i], state2[i-1])
        state2[i] = max(state2[i-1], state1[i-1] - arr[i])
    
    return max(state1[n-1], state2[n-1])
```

### 📌 Step-by-Step Walkthrough

**Example 1: Climbing Stairs (n=5)**

```
Can climb 1 or 2 steps at a time.
How many ways to reach top?

dp[i] = ways to reach step i

Base cases:
  dp[0] = 1 (one way: stay at ground)
  dp[1] = 1 (one way: one step)

Recurrence:
  dp[i] = dp[i-1] + dp[i-2]
  (come from one step below or two steps below)

i=2: dp[2] = dp[1] + dp[0] = 1 + 1 = 2
i=3: dp[3] = dp[2] + dp[1] = 2 + 1 = 3
i=4: dp[4] = dp[3] + dp[2] = 3 + 2 = 5
i=5: dp[5] = dp[4] + dp[3] = 5 + 3 = 8

Answer: 8 ways
```

**Example 2: House Robber [2,7,9,3,1]**

```
Can't rob adjacent houses.
Maximum money?

dp[i] = max money robbing up to house i

Base cases:
  dp[0] = 2
  dp[1] = max(2, 7) = 7

Recurrence:
  dp[i] = max(dp[i-1], dp[i-2] + arr[i])
  (skip this house OR rob this house + max from i-2)

i=2: dp[2] = max(7, 2+9) = 11
i=3: dp[3] = max(11, 7+3) = 11
i=4: dp[4] = max(11, 11+1) = 12

Answer: 12
```

**Example 3: Coin Change [1,2,5], amount=11**

```
Minimum coins to make amount?

dp[i] = min coins to make amount i

Base case:
  dp[0] = 0 (0 coins for amount 0)

Recurrence:
  dp[i] = min(dp[i-coin] + 1) for all coins

i=1: dp[1] = dp[0]+1 = 1 (use coin 1)
i=2: dp[2] = min(dp[1]+1, dp[0]+1) = 1 (use coin 2)
i=3: dp[3] = min(dp[2]+1, dp[1]+1) = 2
i=4: dp[4] = min(dp[3]+1, dp[2]+1) = 2
i=5: dp[5] = min(dp[4]+1, dp[3]+1, dp[0]+1) = 1
...
i=11: dp[11] = 3 (5+5+1)

Answer: 3 coins
```

### 🧪 Solved Examples

**1. Climbing Stairs**
```python
def climbStairs(n):
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1
```

**2. House Robber**
```python
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2, prev1 = 0, 0
    
    for num in nums:
        current = max(prev1, prev2 + num)
        prev2 = prev1
        prev1 = current
    
    return prev1
```

**3. House Robber II (Circular)**
```python
def rob(nums):
    if len(nums) == 1:
        return nums[0]
    
    def rob_linear(houses):
        prev2, prev1 = 0, 0
        for num in houses:
            current = max(prev1, prev2 + num)
            prev2 = prev1
            prev1 = current
        return prev1
    
    # Either rob first house (exclude last) or rob last house (exclude first)
    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
```

**4. Coin Change**
```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

**5. Coin Change 2 (Count Ways)**
```python
def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]
```

**6. Decode Ways**
```python
def numDecodings(s):
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    
    for i in range(2, n + 1):
        # Single digit
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        
        # Two digits
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]
    
    return dp[n]
```

**7. Jump Game**
```python
def canJump(nums):
    max_reach = 0
    
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
    
    return True
```

**8. Jump Game II (Minimum Jumps)**
```python
def jump(nums):
    jumps = 0
    current_end = 0
    farthest = 0
    
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        
        if i == current_end:
            jumps += 1
            current_end = farthest
    
    return jumps
```

**9. Word Break**
```python
def wordBreak(s, wordDict):
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    word_set = set(wordDict)
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]
```

**10. Longest Increasing Subsequence**
```python
def lengthOfLIS(nums):
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# Optimized O(n log n) with binary search
def lengthOfLIS_optimized(nums):
    tails = []
    
    for num in nums:
        left, right = 0, len(tails)
        
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    
    return len(tails)
```

### ⚠️ Edge Cases & Pitfalls

- **Base cases**: Carefully define dp[0], dp[1]
- **Index out of bounds**: Check i-1, i-2 are valid
- **Empty input**: Handle empty array/string
- **Single element**: May need special handling
- **Integer overflow**: Large DP values
- **Initialization**: Use correct initial values (0, -inf, inf)
- **Order of loops**: Matters for some problems (coin change)

---

## 17.2. 2D DP

---

### ❓ When should I use this?

- State depends on **two variables** (usually two indices)
- Keywords: "grid", "two strings", "matrix", "edit distance"
- `dp[i][j]` represents optimal solution for first i and j elements

### 🧩 Common Problem Types

- Unique paths (grid)
- Minimum path sum
- Longest common subsequence
- Edit distance
- Regular expression matching
- Interleaving string
- Distinct subsequences
- Maximal square

### 🧱 Template (Python)

```python
# Pattern 1: Basic 2D DP (Grid)
def dp_2d_grid(grid):
    rows, cols = len(grid), len(grid[0])
    dp = [[0] * cols for _ in range(rows)]
    
    # Base cases
    dp[0][0] = grid[0][0]
    
    # Fill first row
    for c in range(1, cols):
        dp[0][c] = dp[0][c-1] + grid[0][c]
    
    # Fill first column
    for r in range(1, rows):
        dp[r][0] = dp[r-1][0] + grid[r][0]
    
    # Fill rest of table
    for r in range(1, rows):
        for c in range(1, cols):
            dp[r][c] = compute_from_neighbors(dp, r, c)
    
    return dp[rows-1][cols-1]

# Pattern 2: 2D DP (Two Sequences)
def dp_2d_sequences(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = base_value_i
    for j in range(n + 1):
        dp[0][j] = base_value_j
    
    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + match_value
            else:
                dp[i][j] = compute_from_neighbors(dp, i, j)
    
    return dp[m][n]

# Pattern 3: Space-Optimized 2D DP (1D Array)
def dp_2d_optimized(grid):
    rows, cols = len(grid), len(grid[0])
    dp = [0] * cols
    dp[0] = grid[0][0]
    
    # Initialize first row
    for c in range(1, cols):
        dp[c] = dp[c-1] + grid[0][c]
    
    # Process remaining rows
    for r in range(1, rows):
        dp[0] += grid[r][0]  # Update first column
        
        for c in range(1, cols):
            dp[c] = min(dp[c], dp[c-1]) + grid[r][c]
    
    return dp[cols-1]

# Pattern 4: 2D DP with Path Reconstruction
def dp_2d_with_path(grid):
    rows, cols = len(grid), len(grid[0])
    dp = [[0] * cols for _ in range(rows)]
    parent = [[None] * cols for _ in range(rows)]
    
    # Fill dp table and track parent
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                dp[r][c] = grid[r][c]
            else:
                candidates = []
                if r > 0:
                    candidates.append((dp[r-1][c], 'down'))
                if c > 0:
                    candidates.append((dp[r][c-1], 'right'))
                
                best_val, direction = min(candidates)
                dp[r][c] = best_val + grid[r][c]
                parent[r][c] = direction
    
    # Reconstruct path
    path = []
    r, c = rows - 1, cols - 1
    while parent[r][c]:
        path.append((r, c))
        if parent[r][c] == 'down':
            r -= 1
        else:
            c -= 1
    path.append((0, 0))
    
    return dp[rows-1][cols-1], path[::-1]
```

### 📌 Step-by-Step Walkthrough

**Example 1: Unique Paths (3x3 grid)**

```
Grid: 3 rows, 3 columns
Start: (0,0), End: (2,2)
Can only move right or down

dp[i][j] = number of ways to reach (i,j)

Base cases:
  dp[0][0] = 1
  First row: dp[0][c] = 1 (only one way: all right)
  First col: dp[r][0] = 1 (only one way: all down)

     0  1  2
  0  1  1  1
  1  1  2  3
  2  1  3  6

Recurrence:
  dp[i][j] = dp[i-1][j] + dp[i][j-1]
  (come from above or from left)

dp[1][1] = dp[0][1] + dp[1][0] = 1 + 1 = 2
dp[1][2] = dp[0][2] + dp[1][1] = 1 + 2 = 3
dp[2][1] = dp[1][1] + dp[2][0] = 2 + 1 = 3
dp[2][2] = dp[1][2] + dp[2][1] = 3 + 3 = 6

Answer: 6 paths
```

**Example 2: Longest Common Subsequence "abcde" and "ace"**

```
s1 = "abcde", s2 = "ace"

dp[i][j] = LCS length of s1[0..i-1] and s2[0..j-1]

       ""  a  c  e
    "" 0   0  0  0
    a  0   1  1  1
    b  0   1  1  1
    c  0   1  2  2
    d  0   1  2  2
    e  0   1  2  3

Recurrence:
  if s1[i-1] == s2[j-1]:
    dp[i][j] = dp[i-1][j-1] + 1
  else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

dp[1][1]: 'a'=='a' → dp[0][0]+1 = 1
dp[2][1]: 'b'!='a' → max(dp[1][1], dp[2][0]) = 1
dp[3][2]: 'c'=='c' → dp[2][1]+1 = 2
dp[5][3]: 'e'=='e' → dp[4][2]+1 = 3

Answer: 3 (subsequence "ace")
```

**Example 3: Edit Distance "horse" to "ros"**

```
s1 = "horse", s2 = "ros"

dp[i][j] = min operations to convert s1[0..i-1] to s2[0..j-1]

       ""  r  o  s
    "" 0   1  2  3
    h  1   1  2  3
    o  2   2  1  2
    r  3   2  2  2
    s  4   3  3  2
    e  5   4  4  3

Base cases:
  dp[i][0] = i (delete all i chars)
  dp[0][j] = j (insert all j chars)

Recurrence:
  if s1[i-1] == s2[j-1]:
    dp[i][j] = dp[i-1][j-1]
  else:
    dp[i][j] = 1 + min(
      dp[i-1][j],    # delete
      dp[i][j-1],    # insert
      dp[i-1][j-1]   # replace
    )

Answer: 3 operations
```

### 🧪 Solved Examples

**1. Unique Paths**
```python
def uniquePaths(m, n):
    dp = [[1] * n for _ in range(m)]
    
    for r in range(1, m):
        for c in range(1, n):
            dp[r][c] = dp[r-1][c] + dp[r][c-1]
    
    return dp[m-1][n-1]

# Space optimized
def uniquePaths_optimized(m, n):
    dp = [1] * n
    
    for r in range(1, m):
        for c in range(1, n):
            dp[c] += dp[c-1]
    
    return dp[n-1]
```

**2. Unique Paths II (With Obstacles)**
```python
def uniquePathsWithObstacles(grid):
    if not grid or grid[0][0] == 1:
        return 0
    
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    
    for r in range(m):
        for c in range(n):
            if grid[r][c] == 1:
                dp[r][c] = 0
            elif r == 0 and c == 0:
                continue
            else:
                if r > 0:
                    dp[r][c] += dp[r-1][c]
                if c > 0:
                    dp[r][c] += dp[r][c-1]
    
    return dp[m-1][n-1]
```

**3. Minimum Path Sum**
```python
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    
    # First row
    for c in range(1, n):
        dp[0][c] = dp[0][c-1] + grid[0][c]
    
    # First column
    for r in range(1, m):
        dp[r][0] = dp[r-1][0] + grid[r][0]
    
    # Rest of grid
    for r in range(1, m):
        for c in range(1, n):
            dp[r][c] = min(dp[r-1][c], dp[r][c-1]) + grid[r][c]
    
    return dp[m-1][n-1]
```

**4. Longest Common Subsequence**
```python
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

**5. Edit Distance**
```python
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # delete
                    dp[i][j-1],    # insert
                    dp[i-1][j-1]   # replace
                )
    
    return dp[m][n]
```

**6. Maximal Square**
```python
def maximalSquare(matrix):
    if not matrix:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    dp = [[0] * cols for _ in range(rows)]
    max_side = 0
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == '1':
                if r == 0 or c == 0:
                    dp[r][c] = 1
                else:
                    dp[r][c] = min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1]) + 1
                max_side = max(max_side, dp[r][c])
    
    return max_side * max_side
```

**7. Distinct Subsequences**
```python
def numDistinct(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Empty string t can be formed in one way
    for i in range(m + 1):
        dp[i][0] = 1
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
            else:
                dp[i][j] = dp[i-1][j]
    
    return dp[m][n]
```

**8. Interleaving String**
```python
def isInterleave(s1, s2, s3):
    m, n = len(s1), len(s2)
    
    if m + n != len(s3):
        return False
    
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # First row
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
    
    # First column
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or
                (dp[i][j-1] and s2[j-1] == s3[i+j-1])
            )
    
    return dp[m][n]
```

**9. Regular Expression Matching**
```python
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Handle patterns like a*, a*b*, etc.
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # Match zero occurrences
                dp[i][j] = dp[i][j-2]
                
                # Match one or more occurrences
                if p[j-2] == s[i-1] or p[j-2] == '.':
                    dp[i][j] = dp[i][j] or dp[i-1][j]
            elif p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
    
    return dp[m][n]
```

**10. Wildcard Matching**
```python
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Handle leading '*'
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # Match empty or match one char
                dp[i][j] = dp[i][j-1] or dp[i-1][j]
            elif p[j-1] == '?' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
    
    return dp[m][n]
```

### ⚠️ Edge Cases & Pitfalls

- **Base cases**: Initialize first row and column correctly
- **Index confusion**: dp[i][j] often represents s1[0..i-1] and s2[0..j-1]
- **Empty strings**: Handle m=0 or n=0
- **Space optimization**: Can reduce 2D to 1D for many problems
- **Order of iteration**: Must fill table in correct order
- **Off-by-one errors**: Careful with string indexing

---

## 17.3. Knapsack Patterns

---

### ❓ When should I use this?

- **Optimization problem** with capacity constraint
- Keywords: "subset", "target sum", "capacity", "weight/value"
- Choose items to maximize/minimize objective

**Types:**
1. **0/1 Knapsack**: Each item used once
2. **Unbounded Knapsack**: Items can be reused
3. **Bounded Knapsack**: Each item has limited quantity

### 🧩 Common Problem Types

- 0/1 Knapsack
- Partition equal subset sum
- Target sum
- Coin change (unbounded)
- Combination sum IV
- Perfect squares
- Ones and zeroes

### 🧱 Template (Python)

```python
# Pattern 1: 0/1 Knapsack (2D DP)
def knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i-1
            dp[i][w] = dp[i-1][w]
            
            # Take item i-1 if possible
            if w >= weights[i-1]:
                dp[i][w] = max(dp[i][w], 
                              dp[i-1][w - weights[i-1]] + values[i-1])
    
    return dp[n][capacity]

# Pattern 2: 0/1 Knapsack (1D Space Optimized)
def knapsack_01_optimized(weights, values, capacity):
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        # Iterate backwards to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

# Pattern 3: Unbounded Knapsack
def knapsack_unbounded(weights, values, capacity):
    dp = [0] * (capacity + 1)
    
    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if w >= weights[i]:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

# Pattern 4: Count Ways (Subset Sum)
def count_subsets(nums, target):
    dp = [0] * (target + 1)
    dp[0] = 1  # One way to make sum 0: empty subset
    
    for num in nums:
        for s in range(target, num - 1, -1):
            dp[s] += dp[s - num]
    
    return dp[target]

# Pattern 5: True/False Knapsack (Can Achieve Target)
def can_partition(nums, target):
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for s in range(target, num - 1, -1):
            dp[s] = dp[s] or dp[s - num]
    
    return dp[target]

# Pattern 6: Minimum Items to Reach Target
def min_items(items, target):
    dp = [float('inf')] * (target + 1)
    dp[0] = 0
    
    for t in range(1, target + 1):
        for item in items:
            if t >= item:
                dp[t] = min(dp[t], dp[t - item] + 1)
    
    return dp[target] if dp[target] != float('inf') else -1

# Pattern 7: 2D Knapsack (Two Constraints)
def knapsack_2d(items, capacity1, capacity2):
    # items = [(weight1, weight2, value), ...]
    dp = [[0] * (capacity2 + 1) for _ in range(capacity1 + 1)]
    
    for w1, w2, val in items:
        for c1 in range(capacity1, w1 - 1, -1):
            for c2 in range(capacity2, w2 - 1, -1):
                dp[c1][c2] = max(dp[c1][c2], 
                                dp[c1 - w1][c2 - w2] + val)
    
    return dp[capacity1][capacity2]
```

### 📌 Step-by-Step Walkthrough

**Example 1: 0/1 Knapsack**

```
weights = [1, 2, 3], values = [6, 10, 12], capacity = 5

dp[i][w] = max value using first i items with capacity w

     w=0  1  2  3  4  5
i=0   0   0  0  0  0  0
i=1   0   6  6  6  6  6  (item 0: w=1, v=6)
i=2   0   6 10 16 16 16  (item 1: w=2, v=10)
i=3   0   6 10 16 18 22  (item 2: w=3, v=12)

At i=1, w=1: Take item 0 → dp[1][1] = 6
At i=2, w=3: Take both items 0,1 → dp[2][3] = 16
At i=3, w=5: Take items 1,2 → dp[3][5] = 22

Answer: 22
```

**Example 2: Partition Equal Subset Sum [1,5,11,5]**

```
Total sum = 22, target = 11

dp[s] = can we make sum s?

Initial: dp[0] = True

Process 1: dp[1] = True
Process 5: dp[5] = True, dp[6] = True
Process 11: dp[11] = True, dp[16] = True, dp[12] = True, dp[17] = True
Process 5: dp[11] remains True (already True)

dp[11] = True → Can partition

Subsets: {1,5,5} and {11}
```

### 🧪 Solved Examples

**1. 0/1 Knapsack**
```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]
            if w >= weights[i-1]:
                dp[i][w] = max(dp[i][w], 
                              dp[i-1][w - weights[i-1]] + values[i-1])
    
    return dp[n][capacity]
```

**2. Partition Equal Subset Sum**
```python
def canPartition(nums):
    total = sum(nums)
    if total % 2:
        return False
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for s in range(target, num - 1, -1):
            dp[s] = dp[s] or dp[s - num]
    
    return dp[target]
```

**3. Target Sum**
```python
def findTargetSumWays(nums, target):
    total = sum(nums)
    if abs(target) > total or (total + target) % 2:
        return 0
    
    # Convert to subset sum problem
    # sum(P) - sum(N) = target
    # sum(P) + sum(N) = total
    # 2*sum(P) = target + total
    # sum(P) = (target + total) / 2
    
    subset_sum = (target + total) // 2
    dp = [0] * (subset_sum + 1)
    dp[0] = 1
    
    for num in nums:
        for s in range(subset_sum, num - 1, -1):
            dp[s] += dp[s - num]
    
    return dp[subset_sum]
```

**4. Coin Change (Unbounded Knapsack)**
```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] = min(dp[a], dp[a - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

**5. Coin Change 2 (Count Ways)**
```python
def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] += dp[a - coin]
    
    return dp[amount]
```

**6. Perfect Squares**
```python
def numSquares(n):
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    squares = [i*i for i in range(1, int(n**0.5) + 1)]
    
    for i in range(1, n + 1):
        for square in squares:
            if i >= square:
                dp[i] = min(dp[i], dp[i - square] + 1)
    
    return dp[n]
```

**7. Ones and Zeroes (2D Knapsack)**
```python
def findMaxForm(strs, m, n):
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for s in strs:
        zeros = s.count('0')
        ones = s.count('1')
        
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1):
                dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
    
    return dp[m][n]
```

**8. Last Stone Weight II**
```python
def lastStoneWeightII(stones):
    total = sum(stones)
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for stone in stones:
        for s in range(target, stone - 1, -1):
            dp[s] = dp[s] or dp[s - stone]
    
    # Find largest sum <= target
    for s in range(target, -1, -1):
        if dp[s]:
            return total - 2 * s
    
    return 0
```

### ⚠️ Edge Cases & Pitfalls

- **0/1 vs Unbounded**: Iterate backwards for 0/1, forwards for unbounded
- **Target larger than sum**: Impossible to achieve
- **Negative numbers**: Some problems don't allow negatives
- **Zero values**: Handle items with zero weight/value
- **Integer overflow**: Large sums may overflow
- **Space optimization**: 2D → 1D requires backward iteration for 0/1

---

## 17.4. Subsequence / Subarray DP

---

### ❓ When should I use this?

- Problems involving **contiguous subarrays** or **subsequences**
- Keywords: "subarray", "subsequence", "substring", "consecutive"
- **Subarray**: Contiguous elements
- **Subsequence**: Not necessarily contiguous, but maintain order

### 🧩 Common Problem Types

**Subarray:**
- Maximum subarray sum (Kadane's)
- Maximum product subarray
- Longest turbulent subarray
- Subarray sum equals K

**Subsequence:**
- Longest increasing subsequence
- Longest common subsequence
- Longest palindromic subsequence
- Number of longest increasing subsequence

### 🧱 Template (Python)

```python
# Pattern 1: Maximum Subarray (Kadane's Algorithm)
def max_subarray(nums):
    max_sum = nums[0]
    current_sum = nums[0]
    
    for i in range(1, len(nums)):
        # Either extend current subarray or start new one
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Pattern 2: Longest Increasing Subsequence
def lis(nums):
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # dp[i] = LIS ending at i
    
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# Pattern 3: Longest Common Subsequence
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# Pattern 4: Longest Palindromic Subsequence
def lps(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    
    # Single character palindromes
    for i in range(n):
        dp[i][i] = 1
    
    # Build up from length 2
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    
    return dp[0][n-1]

# Pattern 5: Subarray with Target Sum
def subarray_sum(nums, k):
    count = 0
    current_sum = 0
    sum_count = {0: 1}  # prefix sum -> count
    
    for num in nums:
        current_sum += num
        
        # Check if (current_sum - k) exists
        if current_sum - k in sum_count:
            count += sum_count[current_sum - k]
        
        sum_count[current_sum] = sum_count.get(current_sum, 0) + 1
    
    return count

# Pattern 6: Increasing Subsequences (All)
def find_subsequences(nums):
    result = []
    
    def backtrack(start, path):
        if len(path) >= 2:
            result.append(path[:])
        
        used = set()
        for i in range(start, len(nums)):
            # Skip duplicates at same level
            if nums[i] in used:
                continue
            
            # Maintain increasing order
            if not path or nums[i] >= path[-1]:
                used.add(nums[i])
                path.append(nums[i])
                backtrack(i + 1, path)
                path.pop()
    
    backtrack(0, [])
    return result
```

### 🧪 Solved Examples

**1. Maximum Subarray (Kadane's Algorithm)**
```python
def maxSubArray(nums):
    max_sum = current_sum = nums[0]
    
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

**2. Maximum Product Subarray**
```python
def maxProduct(nums):
    if not nums:
        return 0
    
    max_prod = min_prod = result = nums[0]
    
    for num in nums[1:]:
        # Swap if current number is negative
        if num < 0:
            max_prod, min_prod = min_prod, max_prod
        
        max_prod = max(num, max_prod * num)
        min_prod = min(num, min_prod * num)
        
        result = max(result, max_prod)
    
    return result
```

**3. Longest Increasing Subsequence**
```python
def lengthOfLIS(nums):
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# Optimized O(n log n) with binary search
def lengthOfLIS_optimized(nums):
    tails = []
    
    for num in nums:
        left, right = 0, len(tails)
        
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    
    return len(tails)
```

**4. Longest Palindromic Subsequence**
```python
def longestPalindromeSubseq(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    
    for i in range(n):
        dp[i][i] = 1
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    
    return dp[0][n-1]
```

**5. Palindromic Substrings (Count)**
```python
def countSubstrings(s):
    n = len(s)
    count = 0
    
    def expand_around_center(left, right):
        nonlocal count
        while left >= 0 and right < n and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
    
    for i in range(n):
        expand_around_center(i, i)      # Odd length
        expand_around_center(i, i + 1)  # Even length
    
    return count
```

**6. Longest Palindromic Substring**
```python
def longestPalindrome(s):
    if not s:
        return ""
    
    start = 0
    max_len = 0
    
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    for i in range(len(s)):
        len1 = expand_around_center(i, i)
        len2 = expand_around_center(i, i + 1)
        length = max(len1, len2)
        
        if length > max_len:
            max_len = length
            start = i - (length - 1) // 2
    
    return s[start:start + max_len]
```

**7. Number of Longest Increasing Subsequence**
```python
def findNumberOfLIS(nums):
    if not nums:
        return 0
    
    n = len(nums)
    lengths = [1] * n  # Length of LIS ending at i
    counts = [1] * n   # Count of LIS ending at i
    
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                if lengths[j] + 1 > lengths[i]:
                    lengths[i] = lengths[j] + 1
                    counts[i] = counts[j]
                elif lengths[j] + 1 == lengths[i]:
                    counts[i] += counts[j]
    
    max_len = max(lengths)
    return sum(c for l, c in zip(lengths, counts) if l == max_len)
```

**8. Russian Doll Envelopes (2D LIS)**
```python
def maxEnvelopes(envelopes):
    # Sort by width ascending, height descending
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    
    # Find LIS of heights
    tails = []
    
    for _, h in envelopes:
        left, right = 0, len(tails)
        
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < h:
                left = mid + 1
            else:
                right = mid
        
        if left == len(tails):
            tails.append(h)
        else:
            tails[left] = h
    
    return len(tails)
```

**9. Arithmetic Slices**
```python
def numberOfArithmeticSlices(nums):
    if len(nums) < 3:
        return 0
    
    count = 0
    current = 0
    
    for i in range(2, len(nums)):
        if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
            current += 1
            count += current
        else:
            current = 0
    
    return count
```

**10. Longest Turbulent Subarray**
```python
def maxTurbulenceSize(arr):
    if len(arr) < 2:
        return len(arr)
    
    inc = dec = 1
    max_len = 1
    
    for i in range(1, len(arr)):
        if arr[i] > arr[i-1]:
            inc = dec + 1
            dec = 1
        elif arr[i] < arr[i-1]:
            dec = inc + 1
            inc = 1
        else:
            inc = dec = 1
        
        max_len = max(max_len, inc, dec)
    
    return max_len
```

### ⚠️ Edge Cases & Pitfalls

- **Empty array**: Handle len(nums) == 0
- **Single element**: May need special case
- **All same elements**: No increasing subsequence
- **Negative numbers**: Affects product/sum calculations
- **Palindrome edge cases**: Single char, two chars
- **Subarray vs subsequence**: Don't confuse contiguous vs non-contiguous

---

## 17.5. DP Optimization Techniques

---

### ❓ When should I use this?

- Standard DP is too slow (TLE)
- Space complexity is too high (MLE)
- Need to optimize from O(n²) to O(n) or O(n log n)

### 🧠 Core Optimization Techniques

1. **Space Optimization**: Reduce dimensions (2D → 1D)
2. **Rolling Array**: Keep only last k rows/columns
3. **Monotonic Stack/Queue**: Optimize range queries
4. **Binary Search**: Reduce inner loop from O(n) to O(log n)
5. **Prefix Sum**: Precompute cumulative sums
6. **Divide and Conquer**: Split problem into smaller parts

### 🧱 Template (Python)

```python
# Technique 1: Space Optimization (2D → 1D)
def dp_space_optimized(grid):
    rows, cols = len(grid), len(grid[0])
    dp = [0] * cols
    
    # Initialize first row
    dp[0] = grid[0][0]
    for c in range(1, cols):
        dp[c] = dp[c-1] + grid[0][c]
    
    # Process remaining rows
    for r in range(1, rows):
        dp[0] += grid[r][0]  # Update first column
        
        for c in range(1, cols):
            dp[c] = min(dp[c], dp[c-1]) + grid[r][c]
    
    return dp[cols-1]

# Technique 2: Rolling Array (Keep Last K States)
def dp_rolling_array(arr, k):
    n = len(arr)
    dp = [[0] * n for _ in range(k)]
    
    # Only keep last k rows
    for i in range(n):
        for j in range(n):
            dp[i % k][j] = compute(dp[(i-1) % k], j)
    
    return dp[(n-1) % k]

# Technique 3: LIS with Binary Search (O(n log n))
def lis_optimized(nums):
    tails = []  # tails[i] = smallest tail of LIS of length i+1
    
    for num in nums:
        # Binary search for position
        left, right = 0, len(tails)
        
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    
    return len(tails)

# Technique 4: Prefix Sum Optimization
def range_sum_queries(nums):
    n = len(nums)
    prefix = [0] * (n + 1)
    
    for i in range(n):
        prefix[i+1] = prefix[i] + nums[i]
    
    def query(left, right):
        return prefix[right+1] - prefix[left]
    
    return query

# Technique 5: Monotonic Queue for Sliding Window
def sliding_window_max(nums, k):
    from collections import deque
    
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove elements outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove smaller elements (not useful)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# Technique 6: Divide and Conquer Optimization
def max_subarray_dc(nums, left, right):
    if left == right:
        return nums[left]
    
    mid = (left + right) // 2
    
    # Max in left half
    left_max = max_subarray_dc(nums, left, mid)
    
    # Max in right half
    right_max = max_subarray_dc(nums, mid + 1, right)
    
    # Max crossing mid
    left_sum = float('-inf')
    current_sum = 0
    for i in range(mid, left - 1, -1):
        current_sum += nums[i]
        left_sum = max(left_sum, current_sum)
    
    right_sum = float('-inf')
    current_sum = 0
    for i in range(mid + 1, right + 1):
        current_sum += nums[i]
        right_sum = max(right_sum, current_sum)
    
    cross_max = left_sum + right_sum
    
    return max(left_max, right_max, cross_max)
```

### 🧪 Optimization Examples

**1. Space Optimization: Unique Paths**
```python
# Original: O(m×n) space
def uniquePaths_2d(m, n):
    dp = [[1] * n for _ in range(m)]
    
    for r in range(1, m):
        for c in range(1, n):
            dp[r][c] = dp[r-1][c] + dp[r][c-1]
    
    return dp[m-1][n-1]

# Optimized: O(n) space
def uniquePaths_1d(m, n):
    dp = [1] * n
    
    for r in range(1, m):
        for c in range(1, n):
            dp[c] += dp[c-1]
    
    return dp[n-1]
```

**2. Binary Search Optimization: LIS**
```python
# Original: O(n²)
def lengthOfLIS_n2(nums):
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# Optimized: O(n log n)
def lengthOfLIS_nlogn(nums):
    tails = []
    
    for num in nums:
        import bisect
        pos = bisect.bisect_left(tails, num)
        
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)
```

**3. Monotonic Queue: Sliding Window Maximum**
```python
# Brute force: O(n×k)
def maxSlidingWindow_brute(nums, k):
    result = []
    for i in range(len(nums) - k + 1):
        result.append(max(nums[i:i+k]))
    return result

# Optimized: O(n) with monotonic queue
def maxSlidingWindow_optimized(nums, k):
    from collections import deque
    dq = deque()
    result = []
    
    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

**4. Prefix Sum: Range Sum Query**
```python
class NumArray:
    # Without prefix sum: O(n) per query
    def __init__(self, nums):
        self.nums = nums
    
    def sumRange_slow(self, left, right):
        return sum(self.nums[left:right+1])
    
    # With prefix sum: O(1) per query
    def __init__(self, nums):
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)
    
    def sumRange(self, left, right):
        return self.prefix[right+1] - self.prefix[left]
```

**5. State Reduction: Best Time to Buy/Sell Stock**
```python
# Multiple transactions with cooldown
def maxProfit(prices):
    if not prices:
        return 0
    
    # Three states: hold, sold, rest
    hold = -prices[0]
    sold = 0
    rest = 0
    
    for price in prices[1:]:
        prev_hold = hold
        prev_sold = sold
        prev_rest = rest
        
        hold = max(prev_hold, prev_rest - price)
        sold = prev_hold + price
        rest = max(prev_rest, prev_sold)
    
    return max(sold, rest)
```

### ⚠️ When to Use Each Technique

| Technique | Use When | Example |
|-----------|----------|---------|
| Space optimization | Only need previous row/column | Unique Paths, Min Path Sum |
| Binary search | Inner loop searches sorted array | LIS, Russian Dolls |
| Monotonic queue | Sliding window min/max | Sliding Window Max |
| Prefix sum | Many range queries | Range Sum, Subarray Sum |
| Divide & conquer | Problem splits naturally | Max Subarray |
| State machine | Multiple states at each step | Stock trading |

### 🧠 Interview Tips

- **Identify bottleneck**: "The nested loop makes this O(n²), I can optimize with binary search"
- **Space trade-off**: "I can reduce space from O(m×n) to O(n) by keeping only previous row"
- **Explain optimization**: "Binary search reduces this from O(n) to O(log n) per iteration"
- **Mention alternatives**: "Could also use segment tree, but monotonic queue is simpler here"

**Common follow-ups:**
- "Can you optimize space?" → Show 2D → 1D reduction
- "Can you do better than O(n²)?" → Binary search or other techniques
- "What if there are many queries?" → Preprocessing with prefix sums

### ⏱️ Complexity Improvements

| Problem | Original | Optimized | Technique |
|---------|----------|-----------|-----------|
| LIS | O(n²) | O(n log n) | Binary search |
| Sliding window max | O(nk) | O(n) | Monotonic queue |
| Range sum query | O(n) per query | O(1) per query | Prefix sum |
| Unique paths space | O(m×n) | O(n) | Rolling array |
| Edit distance space | O(m×n) | O(min(m,n)) | Space optimization |

---

**🎯 Key Takeaways for Dynamic Programming:**

1. **Identify DP**: Optimal substructure + overlapping subproblems
2. **Define state**: What does dp[i] or dp[i][j] represent?
3. **Find recurrence**: How to compute current state from previous states?
4. **Base cases**: Initialize correctly
5. **Order matters**: Fill table in correct order
6. **Optimize**: Space reduction, binary search, monotonic structures

---

## 18. Bit Manipulation

---

### ❓ When should I use this?

- Problems involving **binary representation** of numbers
- Keywords: "XOR", "AND", "OR", "bits", "binary", "power of 2", "single number"
- **Space optimization**: Use bits instead of arrays/sets
- **Fast operations**: Bitwise ops are O(1) and very fast
- **Mathematical properties**: XOR tricks, bit counting

**Common scenarios:**
- Find unique element (XOR properties)
- Check/set/clear specific bits
- Count set bits
- Power of 2 checks
- Subset generation
- Bitmask DP

### 🧠 Core Idea (Intuition)

**Bitwise operators:**
- `&` (AND): Both bits must be 1
- `|` (OR): At least one bit is 1
- `^` (XOR): Bits are different
- `~` (NOT): Flip all bits
- `<<` (Left shift): Multiply by 2
- `>>` (Right shift): Divide by 2

**Key properties:**
- `a ^ a = 0` (XOR with itself is 0)
- `a ^ 0 = a` (XOR with 0 is identity)
- `a ^ b ^ a = b` (XOR is commutative and associative)
- `a & (a-1)` removes rightmost set bit
- `a & (-a)` isolates rightmost set bit

**Mental model**: 
- Think of numbers as arrays of bits
- Each bit represents presence/absence of something
- XOR like "toggle switch"

### 🧩 Common Problem Types

- Single number (find unique element)
- Counting bits
- Power of two
- Hamming distance
- Reverse bits
- Bitwise AND of range
- Subset generation with bitmask
- Maximum XOR
- Missing number

### 🧱 Template (Python)

```python
# Pattern 1: Basic Bit Operations
def bit_operations(n):
    # Check if i-th bit is set (0-indexed from right)
    def is_bit_set(n, i):
        return (n & (1 << i)) != 0
    
    # Set i-th bit
    def set_bit(n, i):
        return n | (1 << i)
    
    # Clear i-th bit
    def clear_bit(n, i):
        return n & ~(1 << i)
    
    # Toggle i-th bit
    def toggle_bit(n, i):
        return n ^ (1 << i)
    
    # Get rightmost set bit
    def rightmost_set_bit(n):
        return n & (-n)
    
    # Clear rightmost set bit
    def clear_rightmost_bit(n):
        return n & (n - 1)
    
    # Check if power of 2
    def is_power_of_2(n):
        return n > 0 and (n & (n - 1)) == 0
    
    # Count set bits (Brian Kernighan's algorithm)
    def count_bits(n):
        count = 0
        while n:
            n &= n - 1  # Clear rightmost bit
            count += 1
        return count

# Pattern 2: XOR Tricks
def xor_patterns(nums):
    # Find single number (all others appear twice)
    def single_number(nums):
        result = 0
        for num in nums:
            result ^= num
        return result
    
    # Find two single numbers (all others appear twice)
    def single_numbers_two(nums):
        xor = 0
        for num in nums:
            xor ^= num
        
        # Find rightmost set bit (differentiates the two numbers)
        rightmost = xor & (-xor)
        
        num1 = num2 = 0
        for num in nums:
            if num & rightmost:
                num1 ^= num
            else:
                num2 ^= num
        
        return [num1, num2]
    
    # Find single number (all others appear three times)
    def single_number_three(nums):
        ones = twos = 0
        
        for num in nums:
            twos |= ones & num
            ones ^= num
            threes = ones & twos
            ones &= ~threes
            twos &= ~threes
        
        return ones

# Pattern 3: Bitmask for Subsets
def generate_subsets(nums):
    n = len(nums)
    result = []
    
    # 2^n possible subsets
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    
    return result

# Pattern 4: Bitmask DP
def bitmask_dp(n, graph):
    # Traveling Salesman Problem
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start at city 0
    
    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            
            for v in range(n):
                if mask & (1 << v):
                    continue
                
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], 
                                     dp[mask][u] + graph[u][v])
    
    # Return to start
    final_mask = (1 << n) - 1
    return min(dp[final_mask][i] + graph[i][0] for i in range(n))

# Pattern 5: Bit Counting
def count_bits_range(n):
    # Count bits from 0 to n
    result = [0] * (n + 1)
    
    for i in range(1, n + 1):
        # result[i] = result[i >> 1] + (i & 1)
        result[i] = result[i & (i - 1)] + 1
    
    return result

# Pattern 6: XOR Range Queries
def xor_range(arr):
    n = len(arr)
    prefix_xor = [0] * (n + 1)
    
    for i in range(n):
        prefix_xor[i + 1] = prefix_xor[i] ^ arr[i]
    
    def query(left, right):
        return prefix_xor[right + 1] ^ prefix_xor[left]
    
    return query

# Pattern 7: Gray Code
def gray_code(n):
    result = []
    for i in range(1 << n):
        # Gray code: i ^ (i >> 1)
        result.append(i ^ (i >> 1))
    return result

# Pattern 8: Reverse Bits
def reverse_bits(n):
    result = 0
    for i in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result
```

### 📌 Step-by-Step Walkthrough

**Example 1: Single Number [4,1,2,1,2]**

```
XOR all elements:
4 ^ 1 ^ 2 ^ 1 ^ 2

Binary:
100 ^ 001 ^ 010 ^ 001 ^ 010

Step by step:
4 ^ 1 = 101
101 ^ 2 = 111
111 ^ 1 = 110
110 ^ 2 = 100 = 4

Result: 4 (the unique number)
```

**Example 2: Count Bits from 0 to 5**

```
Number  Binary  Count
0       000     0
1       001     1
2       010     1
3       011     2
4       100     1
5       101     2

Pattern: count[i] = count[i >> 1] + (i & 1)
Or: count[i] = count[i & (i-1)] + 1

i=4 (100): count[4] = count[2] + 0 = 1
i=5 (101): count[5] = count[2] + 1 = 2
```

**Example 3: Generate Subsets of [1,2,3]**

```
n=3, so 2^3 = 8 subsets

mask  Binary  Subset
0     000     []
1     001     [1]
2     010     [2]
3     011     [1,2]
4     100     [3]
5     101     [1,3]
6     110     [2,3]
7     111     [1,2,3]

For mask=5 (101):
  Bit 0 set → include nums[0]=1
  Bit 1 not set
  Bit 2 set → include nums[2]=3
  Subset: [1,3]
```

**Example 4: Two Single Numbers [1,2,1,3,2,5]**

```
XOR all: 1^2^1^3^2^5 = 3^5 = 6 (110 in binary)

Rightmost set bit of 6: bit 1 (010)

Partition by bit 1:
  Group 1 (bit 1 set): 2,3,2 → XOR = 3
  Group 2 (bit 1 not set): 1,1,5 → XOR = 5

Result: [3, 5]
```

### 🧪 Solved Examples

**1. Single Number**
```python
def singleNumber(nums):
    result = 0
    for num in nums:
        result ^= num
    return result
```

**2. Single Number II (Appears 3 Times)**
```python
def singleNumber(nums):
    ones = twos = 0
    
    for num in nums:
        twos |= ones & num
        ones ^= num
        threes = ones & twos
        ones &= ~threes
        twos &= ~threes
    
    return ones
```

**3. Single Number III (Two Unique)**
```python
def singleNumber(nums):
    xor = 0
    for num in nums:
        xor ^= num
    
    # Find rightmost set bit
    rightmost = xor & (-xor)
    
    num1 = num2 = 0
    for num in nums:
        if num & rightmost:
            num1 ^= num
        else:
            num2 ^= num
    
    return [num1, num2]
```

**4. Counting Bits**
```python
def countBits(n):
    result = [0] * (n + 1)
    
    for i in range(1, n + 1):
        result[i] = result[i >> 1] + (i & 1)
    
    return result
```

**5. Number of 1 Bits (Hamming Weight)**
```python
def hammingWeight(n):
    count = 0
    while n:
        n &= n - 1  # Clear rightmost bit
        count += 1
    return count
```

**6. Reverse Bits**
```python
def reverseBits(n):
    result = 0
    for i in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result
```

**7. Power of Two**
```python
def isPowerOfTwo(n):
    return n > 0 and (n & (n - 1)) == 0
```

**8. Power of Four**
```python
def isPowerOfFour(n):
    # Must be power of 2 and bit set at even position
    return n > 0 and (n & (n - 1)) == 0 and (n & 0x55555555) != 0
```

**9. Hamming Distance**
```python
def hammingDistance(x, y):
    xor = x ^ y
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count
```

**10. Total Hamming Distance**
```python
def totalHammingDistance(nums):
    total = 0
    n = len(nums)
    
    for i in range(32):
        ones = sum((num >> i) & 1 for num in nums)
        zeros = n - ones
        total += ones * zeros
    
    return total
```

**11. Maximum XOR of Two Numbers**
```python
def findMaximumXOR(nums):
    max_xor = 0
    mask = 0
    
    for i in range(31, -1, -1):
        mask |= (1 << i)
        prefixes = {num & mask for num in nums}
        
        temp = max_xor | (1 << i)
        
        # Check if we can achieve temp
        for prefix in prefixes:
            if temp ^ prefix in prefixes:
                max_xor = temp
                break
    
    return max_xor
```

**12. Missing Number**
```python
def missingNumber(nums):
    result = len(nums)
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result
```

**13. Subsets**
```python
def subsets(nums):
    n = len(nums)
    result = []
    
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    
    return result
```

**14. Gray Code**
```python
def grayCode(n):
    result = []
    for i in range(1 << n):
        result.append(i ^ (i >> 1))
    return result
```

**15. Bitwise AND of Numbers Range**
```python
def rangeBitwiseAnd(left, right):
    shift = 0
    while left < right:
        left >>= 1
        right >>= 1
        shift += 1
    return left << shift
```

### ⚠️ Edge Cases & Pitfalls

- **Signed vs unsigned**: Python handles arbitrary precision, but be careful with 32-bit constraints
- **Negative numbers**: Two's complement representation
- **Overflow**: Shifting beyond bit width
- **Zero**: Special case for many bit operations
- **All bits set**: -1 in two's complement
- **XOR properties**: Remember a ^ a = 0, a ^ 0 = a
- **Bit position**: 0-indexed from right

### ⏱️ Time & Space Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Single bit operation | O(1) | O(1) | Check/set/clear bit |
| Count bits | O(log n) | O(1) | Number of bits |
| XOR all elements | O(n) | O(1) | Linear scan |
| Generate subsets | O(2^n × n) | O(2^n × n) | All subsets |
| Bitmask DP | O(2^n × n²) | O(2^n × n) | TSP-like problems |
| Reverse bits | O(log n) | O(1) | Fixed 32 bits: O(1) |

### 🧠 Interview Tips

- **Explain XOR trick**: "XOR has property that a^a=0, so duplicates cancel out"
- **Bit position**: "I'll check the i-th bit using (n & (1 << i))"
- **Power of 2**: "If n is power of 2, only one bit is set, so n&(n-1)=0"
- **Optimization**: "Bit operations are O(1) and very fast"
- **Bitmask for sets**: "I can use bits to represent presence/absence of elements"

**Common follow-ups:**
- "Can you do it without extra space?" → Bit manipulation often helps
- "What if numbers are very large?" → Discuss bit width constraints
- "Can you optimize with bits?" → Use bitmask instead of set/array

**Red flags to avoid:**
- Not handling negative numbers correctly
- Confusing left shift (multiply) with right shift (divide)
- Off-by-one errors in bit positions
- Not considering integer overflow

---

## 19. Trie (Prefix Tree)

---

### ❓ When should I use this?

- Problems involving **prefix matching** or **string search**
- Keywords: "prefix", "autocomplete", "dictionary", "word search", "longest common prefix"
- Need to store/search **many strings** efficiently
- **Space-time tradeoff**: Use more space for faster search

**Applications:**
- Autocomplete systems
- Spell checkers
- IP routing tables
- Word games (Boggle, Scrabble)
- Longest common prefix

### 🧠 Core Idea (Intuition)

**Trie**: Tree where each node represents a character

**Properties:**
- Root represents empty string
- Path from root to node represents a string
- Each node has children (usually 26 for lowercase letters)
- Mark end of words with flag

**Operations:**
- Insert: O(L) where L = word length
- Search: O(L)
- StartsWith: O(L)

**Mental model**: 
- Like a dictionary organized by first letter, then second letter, etc.
- Each level represents position in word

### 🧩 Common Problem Types

- Implement Trie (insert, search, startsWith)
- Word search II (find all words in board)
- Add and search word (with wildcards)
- Longest word in dictionary
- Replace words (prefix matching)
- Maximum XOR (using binary trie)
- Palindrome pairs

### 🧱 Template (Python)

```python
# Pattern 1: Basic Trie
class TrieNode:
    def __init__(self):
        self.children = {}  # Or [None] * 26 for fixed alphabet
        self.is_end = False
        self.word = None  # Optional: store complete word

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.word = word  # Optional
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def delete(self, word):
        def _delete(node, word, index):
            if index == len(word):
                if not node.is_end:
                    return False
                node.is_end = False
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False
            
            should_delete = _delete(node.children[char], word, index + 1)
            
            if should_delete:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end
            
            return False
        
        _delete(self.root, word, 0)

# Pattern 2: Trie with Wildcard Search
class WordDictionary:
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        def dfs(node, i):
            if i == len(word):
                return node.is_end
            
            char = word[i]
            if char == '.':
                # Try all possible characters
                for child in node.children.values():
                    if dfs(child, i + 1):
                        return True
                return False
            else:
                if char not in node.children:
                    return False
                return dfs(node.children[char], i + 1)
        
        return dfs(self.root, 0)

# Pattern 3: Trie for Word Search II (Backtracking + Trie)
def findWords(board, words):
    # Build trie
    trie = Trie()
    for word in words:
        trie.insert(word)
    
    rows, cols = len(board), len(board[0])
    result = set()
    
    def dfs(r, c, node):
        char = board[r][c]
        if char not in node.children:
            return
        
        next_node = node.children[char]
        
        if next_node.is_end:
            result.add(next_node.word)
        
        # Mark as visited
        board[r][c] = '#'
        
        # Explore 4 directions
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                dfs(nr, nc, next_node)
        
        # Restore
        board[r][c] = char
    
    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie.root)
    
    return list(result)

# Pattern 4: Binary Trie (for XOR problems)
class BinaryTrie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, num):
        node = self.root
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]
    
    def find_max_xor(self, num):
        node = self.root
        max_xor = 0
        
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            # Try to go opposite direction for max XOR
            toggle = 1 - bit
            
            if toggle in node.children:
                max_xor |= (1 << i)
                node = node.children[toggle]
            else:
                node = node.children[bit]
        
        return max_xor

# Pattern 5: Trie with Count (for frequency)
class TrieNodeWithCount:
    def __init__(self):
        self.children = {}
        self.count = 0  # Number of words passing through
        self.end_count = 0  # Number of words ending here

class TrieWithCount:
    def __init__(self):
        self.root = TrieNodeWithCount()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNodeWithCount()
            node = node.children[char]
            node.count += 1
        node.end_count += 1
    
    def count_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.count

# Pattern 6: Trie with DFS (Collect All Words)
def collect_all_words(trie):
    result = []
    
    def dfs(node, path):
        if node.is_end:
            result.append(''.join(path))
        
        for char, child in node.children.items():
            path.append(char)
            dfs(child, path)
            path.pop()
    
    dfs(trie.root, [])
    return result
```

### 📌 Step-by-Step Walkthrough

**Example: Insert ["cat", "car", "card", "dog"]**

```
After inserting "cat":
    root
     |
     c
     |
     a
     |
     t*  (* = end of word)

After inserting "car":
    root
     |
     c
     |
     a
    / \
   t*  r*

After inserting "card":
    root
     |
     c
     |
     a
    / \
   t*  r*
       |
       d*

After inserting "dog":
    root
   /   \
  c     d
  |     |
  a     o
 / \    |
t*  r*  g*
    |
    d*

Search "car": root → c → a → r → is_end=True ✓
Search "ca": root → c → a → is_end=False ✗
StartsWith "ca": root → c → a → exists ✓
```

**Example: Word Search II on Board**

```
Board:
o a a n
e t a e
i h k r
i f l v

Words: ["oath", "pea", "eat", "rain"]

Build trie with all words.

DFS from each cell:
  Start at (0,0) 'o':
    Follow trie path o→a→t→h
    Found "oath" ✓
  
  Start at (1,2) 'a':
    Follow trie path a→t→e (not 'a')
    No match
  
Continue for all cells...

Result: ["oath", "eat"]
```

### 🧪 Solved Examples

**1. Implement Trie**
```python
class Trie:
    def __init__(self):
        self.root = {}
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = True  # End marker
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return '$' in node
    
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True
```

**2. Add and Search Word (with '.')**
```python
class WordDictionary:
    def __init__(self):
        self.root = {}
    
    def addWord(self, word):
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = True
    
    def search(self, word):
        def dfs(node, i):
            if i == len(word):
                return '$' in node
            
            if word[i] == '.':
                for key in node:
                    if key != '$' and dfs(node[key], i + 1):
                        return True
                return False
            else:
                if word[i] not in node:
                    return False
                return dfs(node[word[i]], i + 1)
        
        return dfs(self.root, 0)
```

**3. Word Search II**
```python
def findWords(board, words):
    trie = {}
    for word in words:
        node = trie
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = word
    
    rows, cols = len(board), len(board[0])
    result = set()
    
    def dfs(r, c, node):
        char = board[r][c]
        if char not in node:
            return
        
        next_node = node[char]
        if '$' in next_node:
            result.add(next_node['$'])
        
        board[r][c] = '#'
        
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                dfs(nr, nc, next_node)
        
        board[r][c] = char
    
    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie)
    
    return list(result)
```

**4. Longest Word in Dictionary**
```python
def longestWord(words):
    trie = {}
    for word in words:
        node = trie
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = word
    
    longest = ""
    
    def dfs(node, prefix):
        nonlocal longest
        
        if len(prefix) > len(longest) or (len(prefix) == len(longest) and prefix < longest):
            longest = prefix
        
        for char in node:
            if char != '$' and '$' in node[char]:
                dfs(node[char], prefix + char)
    
    dfs(trie, "")
    return longest
```

**5. Replace Words**
```python
def replaceWords(dictionary, sentence):
    trie = {}
    for word in dictionary:
        node = trie
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = word
    
    def find_root(word):
        node = trie
        for char in word:
            if char not in node:
                return word
            node = node[char]
            if '$' in node:
                return node['$']
        return word
    
    words = sentence.split()
    return ' '.join(find_root(word) for word in words)
```

**6. Maximum XOR of Two Numbers**
```python
def findMaximumXOR(nums):
    trie = {}
    
    # Insert all numbers
    for num in nums:
        node = trie
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]
    
    max_xor = 0
    
    # Find max XOR for each number
    for num in nums:
        node = trie
        current_xor = 0
        
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            toggle = 1 - bit
            
            if toggle in node:
                current_xor |= (1 << i)
                node = node[toggle]
            else:
                node = node[bit]
        
        max_xor = max(max_xor, current_xor)
    
    return max_xor
```

**7. Palindrome Pairs**
```python
def palindromePairs(words):
    def is_palindrome(s):
        return s == s[::-1]
    
    trie = {}
    
    # Build trie with reversed words
    for i, word in enumerate(words):
        node = trie
        for char in reversed(word):
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = i
    
    result = []
    
    for i, word in enumerate(words):
        node = trie
        
        for j, char in enumerate(word):
            # Check if remaining part is palindrome
            if '$' in node and node['$'] != i and is_palindrome(word[j:]):
                result.append([i, node['$']])
            
            if char not in node:
                break
            node = node[char]
        else:
            # Entire word matched
            if '$' in node and node['$'] != i:
                result.append([i, node['$']])
    
    return result
```

### ⚠️ Edge Cases & Pitfalls

- **Empty string**: Handle insertion and search
- **Duplicate words**: Decide if allowed
- **Case sensitivity**: Convert to lowercase if needed
- **Memory**: Trie can use a lot of space
- **Deletion**: Careful not to delete shared prefixes
- **Children representation**: Dict vs array trade-offs
- **End marker**: Don't forget to mark end of words

### ⏱️ Time & Space Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Insert | O(L) | O(L) | L = word length |
| Search | O(L) | O(1) | |
| StartsWith | O(L) | O(1) | |
| Delete | O(L) | O(1) | |
| Build trie | O(N×L) | O(N×L) | N words, avg length L |
| Word search II | O(M×N×4^L) | O(N×L) | M×N board, word length L |

### 🧠 Interview Tips

- **Explain structure**: "Trie is tree where each node represents a character"
- **Space-time tradeoff**: "Trie uses more space but provides fast prefix search"
- **When to use**: "Perfect for autocomplete, spell check, prefix matching"
- **Optimization**: "Can use array of size 26 for lowercase letters, or dict for flexibility"
- **Alternative**: "Could use hashmap, but trie is better for prefix queries"

**Common follow-ups:**
- "How to handle wildcards?" → DFS with backtracking
- "Space optimization?" → Compressed trie (radix tree)
- "What if alphabet is large?" → Use hashmap instead of array
- "Delete operation?" → Recursively remove nodes with no children

**Red flags to avoid:**
- Not marking end of words
- Forgetting to restore board in word search
- Memory issues with large tries
- Not handling empty strings

---

## 20. Segment Tree / Fenwick Tree (Binary Indexed Tree)

---

### ❓ When should I use this?

- Need **range queries** and **point/range updates**
- Keywords: "range sum", "range minimum", "range maximum", "update and query"
- **Segment Tree**: More versatile, supports various operations
- **Fenwick Tree (BIT)**: Simpler, faster, but only for sum/XOR

**Choose:**
- **Segment Tree**: Range min/max, range update, lazy propagation
- **Fenwick Tree**: Range sum, point update (simpler and faster)
- **Prefix Sum**: Only range queries, no updates

### 🧠 Core Idea (Intuition)

**Segment Tree:**
- Binary tree where each node represents a segment
- Leaf nodes are array elements
- Internal nodes store aggregate (sum/min/max) of children
- Build: O(n), Query: O(log n), Update: O(log n)

**Fenwick Tree:**
- Array-based structure using bit manipulation
- Each index stores cumulative sum of certain range
- More space efficient than segment tree
- Build: O(n), Query: O(log n), Update: O(log n)

**Mental model**:
- Segment tree: Like tournament bracket
- Fenwick tree: Like cumulative sum with smart indexing

### 🧩 Common Problem Types

- Range sum query (mutable)
- Range minimum/maximum query
- Count of range sum
- Count of smaller numbers after self
- Reverse pairs
- Skyline problem (with lazy propagation)

### 🧱 Template (Python)

```python
# Pattern 1: Segment Tree (Range Sum)
class SegmentTree:
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (4 * self.n)
        if nums:
            self.build(nums, 0, 0, self.n - 1)
    
    def build(self, nums, node, start, end):
        if start == end:
            self.tree[node] = nums[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self.build(nums, left_child, start, mid)
            self.build(nums, right_child, mid + 1, end)
            
            self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def update(self, index, val):
        self._update(0, 0, self.n - 1, index, val)
    
    def _update(self, node, start, end, index, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if index <= mid:
                self._update(left_child, start, mid, index, val)
            else:
                self._update(right_child, mid + 1, end, index, val)
            
            self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def query(self, left, right):
        return self._query(0, 0, self.n - 1, left, right)
    
    def _query(self, node, start, end, left, right):
        if right < start or left > end:
            return 0  # Out of range
        
        if left <= start and end <= right:
            return self.tree[node]  # Completely in range
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_sum = self._query(left_child, start, mid, left, right)
        right_sum = self._query(right_child, mid + 1, end, left, right)
        
        return left_sum + right_sum

# Pattern 2: Segment Tree (Range Min)
class SegmentTreeMin:
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [float('inf')] * (4 * self.n)
        if nums:
            self.build(nums, 0, 0, self.n - 1)
    
    def build(self, nums, node, start, end):
        if start == end:
            self.tree[node] = nums[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self.build(nums, left_child, start, mid)
            self.build(nums, right_child, mid + 1, end)
            
            self.tree[node] = min(self.tree[left_child], self.tree[right_child])
    
    def query(self, left, right):
        return self._query(0, 0, self.n - 1, left, right)
    
    def _query(self, node, start, end, left, right):
        if right < start or left > end:
            return float('inf')
        
        if left <= start and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_min = self._query(2 * node + 1, start, mid, left, right)
        right_min = self._query(2 * node + 2, mid + 1, end, left, right)
        
        return min(left_min, right_min)

# Pattern 3: Fenwick Tree (Binary Indexed Tree)
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, index, delta):
        # Add delta to element at index (1-indexed)
        index += 1  # Convert to 1-indexed
        while index <= self.n:
            self.tree[index] += delta
            index += index & (-index)  # Add last set bit
    
    def query(self, index):
        # Sum from 0 to index (inclusive, 0-indexed)
        index += 1  # Convert to 1-indexed
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & (-index)  # Remove last set bit
        return result
    
    def range_query(self, left, right):
        # Sum from left to right (inclusive)
        if left == 0:
            return self.query(right)
        return self.query(right) - self.query(left - 1)

# Pattern 4: Fenwick Tree (Build from Array)
class FenwickTreeFromArray:
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (self.n + 1)
        
        for i, num in enumerate(nums):
            self.update(i, num)
    
    def update(self, index, delta):
        index += 1
        while index <= self.n:
            self.tree[index] += delta
            index += index & (-index)
    
    def query(self, index):
        index += 1
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & (-index)
        return result

# Pattern 5: Segment Tree with Lazy Propagation
class SegmentTreeLazy:
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        if nums:
            self.build(nums, 0, 0, self.n - 1)
    
    def build(self, nums, node, start, end):
        if start == end:
            self.tree[node] = nums[start]
        else:
            mid = (start + end) // 2
            self.build(nums, 2 * node + 1, start, mid)
            self.build(nums, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def push_down(self, node, start, end):
        if self.lazy[node] != 0:
            self.tree[node] += (end - start + 1) * self.lazy[node]
            
            if start != end:
                self.lazy[2 * node + 1] += self.lazy[node]
                self.lazy[2 * node + 2] += self.lazy[node]
            
            self.lazy[node] = 0
    
    def update_range(self, left, right, val):
        self._update_range(0, 0, self.n - 1, left, right, val)
    
    def _update_range(self, node, start, end, left, right, val):
        self.push_down(node, start, end)
        
        if right < start or left > end:
            return
        
        if left <= start and end <= right:
            self.lazy[node] += val
            self.push_down(node, start, end)
            return
        
        mid = (start + end) // 2
        self._update_range(2 * node + 1, start, mid, left, right, val)
        self._update_range(2 * node + 2, mid + 1, end, left, right, val)
        
        self.push_down(2 * node + 1, start, mid)
        self.push_down(2 * node + 2, mid + 1, end)
        
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def query(self, left, right):
        return self._query(0, 0, self.n - 1, left, right)
    
    def _query(self, node, start, end, left, right):
        if right < start or left > end:
            return 0
        
        self.push_down(node, start, end)
        
        if left <= start and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_sum = self._query(2 * node + 1, start, mid, left, right)
        right_sum = self._query(2 * node + 2, mid + 1, end, left, right)
        
        return left_sum + right_sum
```

### 📌 Step-by-Step Walkthrough

**Example: Segment Tree for [1,3,5,7,9,11]**

```
Build tree (sum):

Level 0:               36 (sum of all)
                    /      \
Level 1:          9          27
                /   \       /    \
Level 2:       4     5    16     11
              / \   / \   / \
Level 3:     1   3 5   7 9  11

Array indices:  0 1 2 3 4 5
Tree indices:   [36,9,27,4,5,16,11,1,3,5,7,9,11]

Query sum(1, 4): Range [1,4] = 3+5+7+9 = 24
  Start at root (36)
  Left child covers [0,2], partially in range
  Right child covers [3,5], partially in range
  Recursively query both sides
  Return 24

Update index 2 to 6:
  Navigate to leaf at index 2
  Update leaf: 5 → 6
  Propagate up: update all ancestors
  New tree: [37,10,27,4,6,16,11,1,3,6,7,9,11]
```

**Example: Fenwick Tree for [1,3,5,7,9,11]**

```
Build Fenwick tree (1-indexed):

Index:  1  2  3  4  5  6
Value:  1  3  5  7  9  11

Tree:  [0, 1, 4, 5, 16, 9, 20]

tree[1] = 1 (covers index 1)
tree[2] = 1+3 = 4 (covers indices 1-2)
tree[3] = 5 (covers index 3)
tree[4] = 1+3+5+7 = 16 (covers indices 1-4)
tree[5] = 9 (covers index 5)
tree[6] = 9+11 = 20 (covers indices 5-6)

Query sum(0, 3): Sum of first 4 elements
  Start at index 4
  result = tree[4] = 16
  Remove last bit: 4 - (4 & -4) = 0
  Return 16

Update index 2 by +2:
  Start at index 3 (2+1)
  tree[3] += 2
  Add last bit: 3 + (3 & -3) = 4
  tree[4] += 2
  Add last bit: 4 + (4 & -4) = 8 (out of range)
```

### 🧪 Solved Examples

**1. Range Sum Query - Mutable (Segment Tree)**
```python
class NumArray:
    def __init__(self, nums):
        self.n = len(nums)
        self.nums = nums
        self.tree = [0] * (4 * self.n)
        if nums:
            self._build(0, 0, self.n - 1)
    
    def _build(self, node, start, end):
        if start == end:
            self.tree[node] = self.nums[start]
        else:
            mid = (start + end) // 2
            self._build(2 * node + 1, start, mid)
            self._build(2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def update(self, index, val):
        self._update(0, 0, self.n - 1, index, val)
    
    def _update(self, node, start, end, index, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if index <= mid:
                self._update(2 * node + 1, start, mid, index, val)
            else:
                self._update(2 * node + 2, mid + 1, end, index, val)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def sumRange(self, left, right):
        return self._query(0, 0, self.n - 1, left, right)
    
    def _query(self, node, start, end, left, right):
        if right < start or left > end:
            return 0
        if left <= start and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        return (self._query(2 * node + 1, start, mid, left, right) +
                self._query(2 * node + 2, mid + 1, end, left, right))
```

**2. Range Sum Query - Mutable (Fenwick Tree)**
```python
class NumArray:
    def __init__(self, nums):
        self.n = len(nums)
        self.nums = nums[:]
        self.tree = [0] * (self.n + 1)
        
        for i, num in enumerate(nums):
            self._update_tree(i, num)
    
    def _update_tree(self, index, delta):
        index += 1
        while index <= self.n:
            self.tree[index] += delta
            index += index & (-index)
    
    def update(self, index, val):
        delta = val - self.nums[index]
        self.nums[index] = val
        self._update_tree(index, delta)
    
    def _query(self, index):
        index += 1
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & (-index)
        return result
    
    def sumRange(self, left, right):
        return self._query(right) - (self._query(left - 1) if left > 0 else 0)
```

**3. Count of Smaller Numbers After Self**
```python
def countSmaller(nums):
    # Coordinate compression
    sorted_nums = sorted(set(nums))
    rank = {num: i for i, num in enumerate(sorted_nums)}
    
    n = len(sorted_nums)
    tree = [0] * (n + 1)
    
    def update(index):
        index += 1
        while index <= n:
            tree[index] += 1
            index += index & (-index)
    
    def query(index):
        index += 1
        result = 0
        while index > 0:
            result += tree[index]
            index -= index & (-index)
        return result
    
    result = []
    for num in reversed(nums):
        r = rank[num]
        result.append(query(r - 1) if r > 0 else 0)
        update(r)
    
    return result[::-1]
```

**4. Count of Range Sum**
```python
def countRangeSum(nums, lower, upper):
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)
    
    # Coordinate compression
    all_sums = sorted(set(prefix + [s - lower for s in prefix] + [s - upper for s in prefix]))
    rank = {s: i for i, s in enumerate(all_sums)}
    
    n = len(all_sums)
    tree = [0] * (n + 1)
    
    def update(index):
        index += 1
        while index <= n:
            tree[index] += 1
            index += index & (-index)
    
    def query(index):
        index += 1
        result = 0
        while index > 0:
            result += tree[index]
            index -= index & (-index)
        return result
    
    count = 0
    for s in prefix:
        left = rank[s - upper]
        right = rank[s - lower]
        count += query(right) - (query(left - 1) if left > 0 else 0)
        update(rank[s])
    
    return count
```

**5. Reverse Pairs**
```python
def reversePairs(nums):
    def merge_count(nums):
        if len(nums) <= 1:
            return 0
        
        mid = len(nums) // 2
        left = nums[:mid]
        right = nums[mid:]
        
        count = merge_count(left) + merge_count(right)
        
        # Count reverse pairs
        j = 0
        for i in range(len(left)):
            while j < len(right) and left[i] > 2 * right[j]:
                j += 1
            count += j
        
        # Merge
        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                nums[k] = left[i]
                i += 1
            else:
                nums[k] = right[j]
                j += 1
            k += 1
        
        while i < len(left):
            nums[k] = left[i]
            i += 1
            k += 1
        
        while j < len(right):
            nums[k] = right[j]
            j += 1
            k += 1
        
        return count
    
    return merge_count(nums[:])
```

### ⚠️ Edge Cases & Pitfalls

- **1-indexed vs 0-indexed**: Fenwick tree is 1-indexed internally
- **Tree size**: Segment tree needs 4×n space
- **Range bounds**: Check left <= right
- **Empty array**: Handle n=0
- **Integer overflow**: Large sums may overflow
- **Lazy propagation**: Must push down before accessing node
- **Coordinate compression**: Needed for large value ranges

### ⏱️ Time & Space Complexity

| Operation | Segment Tree | Fenwick Tree | Notes |
|-----------|--------------|--------------|-------|
| Build | O(n) | O(n log n) | Fenwick with updates |
| Point update | O(log n) | O(log n) | |
| Range update | O(log n) | - | With lazy propagation |
| Range query | O(log n) | O(log n) | |
| Space | O(4n) = O(n) | O(n) | Fenwick more compact |

### 🧠 Interview Tips

- **Choose structure**: "Fenwick tree is simpler for range sum, segment tree for min/max"
- **Explain indexing**: "Fenwick tree uses 1-indexed array with bit manipulation"
- **Lazy propagation**: "For range updates, I'll use lazy propagation to defer updates"
- **Alternative**: "Could use prefix sum for read-only, but need Fenwick/segment for updates"
- **Coordinate compression**: "For large values, I'll compress coordinates"

**Common follow-ups:**
- "Can you handle range updates?" → Lazy propagation in segment tree
- "What if values are very large?" → Coordinate compression
- "Space optimization?" → Fenwick tree uses less space
- "What about 2D range queries?" → 2D segment tree or Fenwick tree

**Red flags to avoid:**
- Off-by-one errors in indexing
- Not handling tree size correctly (4×n for segment tree)
- Forgetting to push down in lazy propagation
- Confusing Fenwick tree indexing

---

## 21. Advanced Patterns

---

## 21.1. Divide & Conquer

---

### ❓ When should I use this?

- Problem can be **broken into independent subproblems**
- Keywords: "merge", "split", "recursive", "conquer"
- Subproblems are **similar to original** but smaller
- Solutions can be **combined** to solve original problem

**Classic examples:**
- Merge sort, Quick sort
- Binary search
- Maximum subarray (crossing mid)
- Closest pair of points
- Strassen's matrix multiplication

### 🧠 Core Idea (Intuition)

**Three steps:**
1. **Divide**: Break problem into smaller subproblems
2. **Conquer**: Solve subproblems recursively
3. **Combine**: Merge solutions of subproblems

**Mental model**: 
- Like organizing books: split pile in half, sort each half, then merge
- Like tournament: divide players, find winners in each bracket, combine

**Recurrence relation**: T(n) = aT(n/b) + f(n)
- a = number of subproblems
- b = factor by which size reduces
- f(n) = cost to divide and combine

### 🧩 Common Problem Types

- Merge sort / Quick sort
- Maximum subarray (Kadane's alternative)
- Kth largest element (QuickSelect)
- Count of smaller numbers after self
- Reverse pairs
- Different ways to add parentheses
- Median of two sorted arrays

### 🧱 Template (Python)

```python
# Pattern 1: Basic Divide & Conquer
def divide_conquer(arr, left, right):
    # Base case
    if left >= right:
        return base_case(arr[left])
    
    # Divide
    mid = (left + right) // 2
    
    # Conquer
    left_result = divide_conquer(arr, left, mid)
    right_result = divide_conquer(arr, mid + 1, right)
    
    # Combine
    return combine(left_result, right_result)

# Pattern 2: Merge Sort
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Pattern 3: Quick Sort
def quick_sort(arr, left, right):
    if left < right:
        pivot_idx = partition(arr, left, right)
        quick_sort(arr, left, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, right)

def partition(arr, left, right):
    pivot = arr[right]
    i = left - 1
    
    for j in range(left, right):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1

# Pattern 4: QuickSelect (Kth Largest)
def quick_select(arr, k):
    # Find kth smallest (0-indexed)
    def select(left, right, k):
        if left == right:
            return arr[left]
        
        pivot_idx = partition(arr, left, right)
        
        if k == pivot_idx:
            return arr[k]
        elif k < pivot_idx:
            return select(left, pivot_idx - 1, k)
        else:
            return select(pivot_idx + 1, right, k)
    
    return select(0, len(arr) - 1, k)

# Pattern 5: Maximum Subarray (Divide & Conquer)
def max_subarray_dc(arr):
    def helper(left, right):
        if left == right:
            return arr[left]
        
        mid = (left + right) // 2
        
        # Max in left half
        left_max = helper(left, mid)
        # Max in right half
        right_max = helper(mid + 1, right)
        
        # Max crossing mid
        left_sum = float('-inf')
        current = 0
        for i in range(mid, left - 1, -1):
            current += arr[i]
            left_sum = max(left_sum, current)
        
        right_sum = float('-inf')
        current = 0
        for i in range(mid + 1, right + 1):
            current += arr[i]
            right_sum = max(right_sum, current)
        
        cross_max = left_sum + right_sum
        
        return max(left_max, right_max, cross_max)
    
    return helper(0, len(arr) - 1)

# Pattern 6: Count Inversions (Merge Sort variant)
def count_inversions(arr):
    def merge_count(arr):
        if len(arr) <= 1:
            return arr, 0
        
        mid = len(arr) // 2
        left, left_inv = merge_count(arr[:mid])
        right, right_inv = merge_count(arr[mid:])
        
        merged = []
        inv_count = left_inv + right_inv
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inv_count += len(left) - i  # All remaining in left are inversions
                j += 1
        
        merged.extend(left[i:])
        merged.extend(right[j:])
        
        return merged, inv_count
    
    _, count = merge_count(arr)
    return count

# Pattern 7: Closest Pair of Points
def closest_pair(points):
    def distance(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def brute_force(points):
        min_dist = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                min_dist = min(min_dist, distance(points[i], points[j]))
        return min_dist
    
    def closest_in_strip(strip, d):
        min_dist = d
        strip.sort(key=lambda p: p[1])
        
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j][1] - strip[i][1]) < min_dist:
                min_dist = min(min_dist, distance(strip[i], strip[j]))
                j += 1
        
        return min_dist
    
    def helper(px, py):
        n = len(px)
        
        if n <= 3:
            return brute_force(px)
        
        mid = n // 2
        midpoint = px[mid]
        
        pyl = [p for p in py if p[0] <= midpoint[0]]
        pyr = [p for p in py if p[0] > midpoint[0]]
        
        dl = helper(px[:mid], pyl)
        dr = helper(px[mid:], pyr)
        
        d = min(dl, dr)
        
        strip = [p for p in py if abs(p[0] - midpoint[0]) < d]
        
        return min(d, closest_in_strip(strip, d))
    
    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])
    
    return helper(px, py)

# Pattern 8: Different Ways to Add Parentheses
def different_ways(expression):
    memo = {}
    
    def ways(expr):
        if expr in memo:
            return memo[expr]
        
        # Base case: single number
        if expr.isdigit():
            return [int(expr)]
        
        result = []
        
        for i, char in enumerate(expr):
            if char in '+-*':
                # Divide
                left = ways(expr[:i])
                right = ways(expr[i+1:])
                
                # Combine
                for l in left:
                    for r in right:
                        if char == '+':
                            result.append(l + r)
                        elif char == '-':
                            result.append(l - r)
                        else:
                            result.append(l * r)
        
        memo[expr] = result
        return result
    
    return ways(expression)
```

### 🧪 Solved Examples

**1. Merge Sort**
```python
def sortArray(nums):
    if len(nums) <= 1:
        return nums
    
    mid = len(nums) // 2
    left = sortArray(nums[:mid])
    right = sortArray(nums[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

**2. Kth Largest Element (QuickSelect)**
```python
def findKthLargest(nums, k):
    k = len(nums) - k  # Convert to kth smallest (0-indexed)
    
    def select(left, right):
        pivot = nums[right]
        i = left
        
        for j in range(left, right):
            if nums[j] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        
        nums[i], nums[right] = nums[right], nums[i]
        
        if i == k:
            return nums[i]
        elif i < k:
            return select(i + 1, right)
        else:
            return select(left, i - 1)
    
    return select(0, len(nums) - 1)
```

**3. Maximum Subarray (Divide & Conquer)**
```python
def maxSubArray(nums):
    def helper(left, right):
        if left == right:
            return nums[left]
        
        mid = (left + right) // 2
        
        left_max = helper(left, mid)
        right_max = helper(mid + 1, right)
        
        # Cross max
        left_sum = float('-inf')
        current = 0
        for i in range(mid, left - 1, -1):
            current += nums[i]
            left_sum = max(left_sum, current)
        
        right_sum = float('-inf')
        current = 0
        for i in range(mid + 1, right + 1):
            current += nums[i]
            right_sum = max(right_sum, current)
        
        cross_max = left_sum + right_sum
        
        return max(left_max, right_max, cross_max)
    
    return helper(0, len(nums) - 1)
```

**4. Different Ways to Add Parentheses**
```python
def diffWaysToCompute(expression):
    memo = {}
    
    def compute(expr):
        if expr in memo:
            return memo[expr]
        
        if expr.isdigit():
            return [int(expr)]
        
        result = []
        
        for i, char in enumerate(expr):
            if char in '+-*':
                left = compute(expr[:i])
                right = compute(expr[i+1:])
                
                for l in left:
                    for r in right:
                        if char == '+':
                            result.append(l + r)
                        elif char == '-':
                            result.append(l - r)
                        else:
                            result.append(l * r)
        
        memo[expr] = result
        return result
    
    return compute(expression)
```

**5. Median of Two Sorted Arrays**
```python
def findMedianSortedArrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        
        maxLeft1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        minRight1 = float('inf') if partition1 == m else nums1[partition1]
        
        maxLeft2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        minRight2 = float('inf') if partition2 == n else nums2[partition2]
        
        if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
            if (m + n) % 2 == 0:
                return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2
            else:
                return max(maxLeft1, maxLeft2)
        elif maxLeft1 > minRight2:
            right = partition1 - 1
        else:
            left = partition1 + 1
    
    return 0
```

### ⚠️ Edge Cases & Pitfalls

- **Base case**: Must handle single element or empty array
- **Stack overflow**: Very deep recursion for large inputs
- **Partition choice**: Poor pivot in QuickSort leads to O(n²)
- **Combining step**: Often the tricky part
- **Index bounds**: Careful with left, mid, right
- **Stability**: Merge sort is stable, quick sort is not

### ⏱️ Time & Space Complexity

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| QuickSelect | O(n) | O(n) | O(n²) | O(log n) | - |
| Max Subarray | O(n log n) | O(n log n) | O(n log n) | O(log n) | - |

---

## 21.2. Sweep Line

---

### ❓ When should I use this?

- Problems involving **intervals** or **events** on a line
- Keywords: "intervals", "meetings", "events", "skyline", "overlap"
- Need to process **events in order** (start/end times)
- Track **active intervals** at any point

**Applications:**
- Meeting rooms
- Skyline problem
- Merge intervals
- Insert interval
- Calendar scheduling

### 🧠 Core Idea (Intuition)

**Sweep line**: Imagine a vertical line sweeping from left to right

**Algorithm:**
1. Create events for interval starts and ends
2. Sort events by position
3. Process events in order, updating state
4. Track active intervals or count

**Mental model**: 
- Like watching cars enter/exit parking lot
- Count cars inside at any time

### 🧩 Common Problem Types

- Meeting rooms II (minimum rooms)
- Merge intervals
- Insert interval
- Skyline problem
- My calendar I/II/III
- Employee free time
- Remove covered intervals

### 🧱 Template (Python)

```python
# Pattern 1: Sweep Line with Events
def sweep_line_events(intervals):
    events = []
    
    for start, end in intervals:
        events.append((start, 1))   # Start event
        events.append((end, -1))    # End event
    
    events.sort()
    
    active = 0
    max_active = 0
    
    for time, delta in events:
        active += delta
        max_active = max(max_active, active)
    
    return max_active

# Pattern 2: Sweep Line for Merging Intervals
def merge_intervals(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            # Overlapping, merge
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            # Non-overlapping, add new interval
            merged.append(current)
    
    return merged

# Pattern 3: Sweep Line with Priority Queue (Skyline)
def skyline_problem(buildings):
    import heapq
    
    # Create events
    events = []
    for left, right, height in buildings:
        events.append((left, -height, right))   # Start (negative height for max heap)
        events.append((right, 0, 0))            # End
    
    events.sort()
    
    result = []
    heap = [(0, float('inf'))]  # (height, end_time)
    
    for x, neg_h, r in events:
        # Remove buildings that have ended
        while heap[0][1] <= x:
            heapq.heappop(heap)
        
        if neg_h:  # Start event
            heapq.heappush(heap, (neg_h, r))
        
        # Current max height
        max_h = -heap[0][0]
        
        if not result or result[-1][1] != max_h:
            result.append([x, max_h])
    
    return result

# Pattern 4: Difference Array (Range Updates)
def range_updates(n, updates):
    diff = [0] * (n + 1)
    
    for start, end, val in updates:
        diff[start] += val
        diff[end + 1] -= val
    
    # Compute prefix sum to get actual values
    result = []
    current = 0
    for i in range(n):
        current += diff[i]
        result.append(current)
    
    return result

# Pattern 5: Sweep Line with Count
def count_overlaps(intervals):
    events = []
    
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    
    events.sort(key=lambda x: (x[0], -x[1]))  # End before start at same time
    
    count = 0
    max_overlap = 0
    
    for time, delta in events:
        count += delta
        max_overlap = max(max_overlap, count)
    
    return max_overlap

# Pattern 6: Insert Interval
def insert_interval(intervals, new_interval):
    result = []
    i = 0
    n = len(intervals)
    
    # Add all intervals before new_interval
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge overlapping intervals
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    
    result.append(new_interval)
    
    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result
```

### 🧪 Solved Examples

**1. Meeting Rooms II (Minimum Rooms)**
```python
def minMeetingRooms(intervals):
    if not intervals:
        return 0
    
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    
    events.sort(key=lambda x: (x[0], x[1]))
    
    rooms = 0
    max_rooms = 0
    
    for time, delta in events:
        rooms += delta
        max_rooms = max(max_rooms, rooms)
    
    return max_rooms
```

**2. Merge Intervals**
```python
def merge(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)
    
    return merged
```

**3. Insert Interval**
```python
def insert(intervals, newInterval):
    result = []
    i = 0
    
    # Before new interval
    while i < len(intervals) and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge overlapping
    while i < len(intervals) and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    
    result.append(newInterval)
    
    # After new interval
    while i < len(intervals):
        result.append(intervals[i])
        i += 1
    
    return result
```

**4. The Skyline Problem**
```python
def getSkyline(buildings):
    import heapq
    
    events = []
    for left, right, height in buildings:
        events.append((left, -height, right))
        events.append((right, 0, 0))
    
    events.sort()
    
    result = []
    heap = [(0, float('inf'))]
    
    for x, neg_h, r in events:
        while heap[0][1] <= x:
            heapq.heappop(heap)
        
        if neg_h:
            heapq.heappush(heap, (neg_h, r))
        
        max_h = -heap[0][0]
        
        if not result or result[-1][1] != max_h:
            result.append([x, max_h])
    
    return result
```

**5. My Calendar I**
```python
class MyCalendar:
    def __init__(self):
        self.events = []
    
    def book(self, start, end):
        for s, e in self.events:
            if start < e and end > s:
                return False
        
        self.events.append((start, end))
        return True
```

**6. My Calendar II (Double Booking)**
```python
class MyCalendarTwo:
    def __init__(self):
        self.events = []
        self.overlaps = []
    
    def book(self, start, end):
        # Check if triple booking
        for s, e in self.overlaps:
            if start < e and end > s:
                return False
        
        # Add new overlaps
        for s, e in self.events:
            if start < e and end > s:
                self.overlaps.append((max(start, s), min(end, e)))
        
        self.events.append((start, end))
        return True
```

**7. Employee Free Time**
```python
def employeeFreeTime(schedule):
    events = []
    
    for employee in schedule:
        for interval in employee:
            events.append((interval.start, 1))
            events.append((interval.end, -1))
    
    events.sort()
    
    result = []
    count = 0
    prev_time = None
    
    for time, delta in events:
        if count == 0 and prev_time is not None:
            result.append(Interval(prev_time, time))
        
        count += delta
        prev_time = time
    
    return result
```

**8. Remove Covered Intervals**
```python
def removeCoveredIntervals(intervals):
    intervals.sort(key=lambda x: (x[0], -x[1]))
    
    count = 0
    prev_end = 0
    
    for start, end in intervals:
        if end > prev_end:
            count += 1
            prev_end = end
    
    return count
```

### ⚠️ Edge Cases & Pitfalls

- **Sorting key**: End events should process before start at same time
- **Boundary conditions**: Inclusive vs exclusive intervals
- **Empty intervals**: Handle zero-length intervals
- **Overlapping vs touching**: [1,2] and [2,3] may or may not overlap
- **Event ordering**: Critical for correctness

### ⏱️ Time & Space Complexity

| Problem | Time | Space | Notes |
|---------|------|-------|-------|
| Meeting rooms | O(n log n) | O(n) | Sorting events |
| Merge intervals | O(n log n) | O(n) | Sorting |
| Skyline | O(n log n) | O(n) | With heap |
| Insert interval | O(n) | O(n) | Linear scan |
| Calendar | O(n²) | O(n) | Check all events |

---

## 21.3. Interval Problems

---

### ❓ When should I use this?

- Problems with **time ranges** or **segments**
- Keywords: "intervals", "ranges", "overlapping", "merge", "intersect"
- Need to find **overlaps**, **gaps**, or **coverage**

### 🧩 Common Problem Types

- Merge intervals
- Insert interval
- Interval list intersections
- Non-overlapping intervals
- Minimum arrows to burst balloons
- Data stream as disjoint intervals

### 🧱 Template (Python)

```python
# Pattern 1: Check Overlap
def is_overlap(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    return start1 < end2 and start2 < end1

# Pattern 2: Merge Two Intervals
def merge_two(interval1, interval2):
    if not is_overlap(interval1, interval2):
        return None
    return [min(interval1[0], interval2[0]), max(interval1[1], interval2[1])]

# Pattern 3: Interval Intersection
def intersect(interval1, interval2):
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    
    if start < end:
        return [start, end]
    return None

# Pattern 4: Sort and Merge
def merge_intervals(intervals):
    if not intervals:
        return []
    
    intervals.sort()
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)
    
    return merged

# Pattern 5: Two Pointer for Intersections
def interval_intersections(list1, list2):
    result = []
    i = j = 0
    
    while i < len(list1) and j < len(list2):
        start = max(list1[i][0], list2[j][0])
        end = min(list1[i][1], list2[j][1])
        
        if start <= end:
            result.append([start, end])
        
        # Move pointer of interval that ends first
        if list1[i][1] < list2[j][1]:
            i += 1
        else:
            j += 1
    
    return result

# Pattern 6: Greedy for Non-overlapping
def erase_overlap_intervals(intervals):
    if not intervals:
        return 0
    
    intervals.sort(key=lambda x: x[1])  # Sort by end time
    
    count = 0
    end = float('-inf')
    
    for start, curr_end in intervals:
        if start >= end:
            end = curr_end
        else:
            count += 1
    
    return count
```

### 🧪 Solved Examples

**1. Merge Intervals**
```python
def merge(intervals):
    intervals.sort()
    merged = [intervals[0]]
    
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    
    return merged
```

**2. Interval List Intersections**
```python
def intervalIntersection(firstList, secondList):
    result = []
    i = j = 0
    
    while i < len(firstList) and j < len(secondList):
        start = max(firstList[i][0], secondList[j][0])
        end = min(firstList[i][1], secondList[j][1])
        
        if start <= end:
            result.append([start, end])
        
        if firstList[i][1] < secondList[j][1]:
            i += 1
        else:
            j += 1
    
    return result
```

**3. Non-overlapping Intervals**
```python
def eraseOverlapIntervals(intervals):
    intervals.sort(key=lambda x: x[1])
    
    count = 0
    end = float('-inf')
    
    for start, curr_end in intervals:
        if start >= end:
            end = curr_end
        else:
            count += 1
    
    return count
```

**4. Minimum Number of Arrows**
```python
def findMinArrowShots(points):
    points.sort(key=lambda x: x[1])
    
    arrows = 0
    end = float('-inf')
    
    for start, curr_end in points:
        if start > end:
            arrows += 1
            end = curr_end
    
    return arrows
```

**5. Data Stream as Disjoint Intervals**
```python
class SummaryRanges:
    def __init__(self):
        self.intervals = []
    
    def addNum(self, value):
        new_interval = [value, value]
        result = []
        i = 0
        
        # Add intervals before
        while i < len(self.intervals) and self.intervals[i][1] < value - 1:
            result.append(self.intervals[i])
            i += 1
        
        # Merge overlapping
        while i < len(self.intervals) and self.intervals[i][0] <= value + 1:
            new_interval[0] = min(new_interval[0], self.intervals[i][0])
            new_interval[1] = max(new_interval[1], self.intervals[i][1])
            i += 1
        
        result.append(new_interval)
        
        # Add intervals after
        while i < len(self.intervals):
            result.append(self.intervals[i])
            i += 1
        
        self.intervals = result
    
    def getIntervals(self):
        return self.intervals
```

---

## 21.4. Monotonic Stack

---

### ❓ When should I use this?

- Need to find **next greater/smaller** element
- Keywords: "next greater", "next smaller", "previous", "histogram", "temperature"
- Maintain elements in **monotonic order** (increasing or decreasing)
- Often involves **looking left or right** for comparison

### 🧠 Core Idea (Intuition)

**Monotonic stack**: Stack that maintains elements in monotonic order

**Two types:**
1. **Monotonic increasing**: Elements increase from bottom to top
2. **Monotonic decreasing**: Elements decrease from bottom to top

**When to pop:**
- Increasing stack: Pop when current element is smaller
- Decreasing stack: Pop when current element is larger

**Mental model**: 
- Like a line of people where you can only see taller/shorter ones ahead
- Stack remembers potential candidates

### 🧩 Common Problem Types

- Next greater element
- Daily temperatures
- Largest rectangle in histogram
- Trapping rain water
- Remove K digits
- Sum of subarray minimums
- Online stock span

### 🧱 Template (Python)

```python
# Pattern 1: Next Greater Element (Monotonic Decreasing Stack)
def next_greater_element(nums):
    n = len(nums)
    result = [-1] * n
    stack = []  # Store indices
    
    for i in range(n):
        # Pop smaller elements
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        
        stack.append(i)
    
    return result

# Pattern 2: Next Smaller Element (Monotonic Increasing Stack)
def next_smaller_element(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    
    for i in range(n):
        # Pop larger elements
        while stack and nums[stack[-1]] > nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        
        stack.append(i)
    
    return result

# Pattern 3: Previous Greater Element
def previous_greater_element(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    
    for i in range(n):
        # Pop smaller or equal elements
        while stack and nums[stack[-1]] <= nums[i]:
            stack.pop()
        
        if stack:
            result[i] = nums[stack[-1]]
        
        stack.append(i)
    
    return result

# Pattern 4: Largest Rectangle in Histogram
def largest_rectangle(heights):
    stack = []
    max_area = 0
    heights.append(0)  # Sentinel to flush stack
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        stack.append(i)
    
    return max_area

# Pattern 5: Remove K Digits (Greedy + Monotonic Stack)
def remove_k_digits(num, k):
    stack = []
    
    for digit in num:
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        
        stack.append(digit)
    
    # Remove remaining k digits
    stack = stack[:-k] if k > 0 else stack
    
    # Remove leading zeros
    result = ''.join(stack).lstrip('0')
    
    return result if result else '0'

# Pattern 6: Sum of Subarray Minimums
def sum_subarray_mins(arr):
    MOD = 10**9 + 7
    n = len(arr)
    
    # Find previous less element
    left = [-1] * n
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] > arr[i]:
            stack.pop()
        left[i] = stack[-1] if stack else -1
        stack.append(i)
    
    # Find next less element
    right = [n] * n
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()
        right[i] = stack[-1] if stack else n
        stack.append(i)
    
    # Calculate sum
    result = 0
    for i in range(n):
        left_count = i - left[i]
        right_count = right[i] - i
        result = (result + arr[i] * left_count * right_count) % MOD
    
    return result
```

### 🧪 Solved Examples

**1. Daily Temperatures**
```python
def dailyTemperatures(temperatures):
    n = len(temperatures)
    result = [0] * n
    stack = []
    
    for i in range(n):
        while stack and temperatures[stack[-1]] < temperatures[i]:
            idx = stack.pop()
            result[idx] = i - idx
        
        stack.append(i)
    
    return result
```

**2. Next Greater Element I**
```python
def nextGreaterElement(nums1, nums2):
    next_greater = {}
    stack = []
    
    for num in nums2:
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    
    return [next_greater.get(num, -1) for num in nums1]
```

**3. Next Greater Element II (Circular)**
```python
def nextGreaterElements(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    
    # Process array twice for circular
    for i in range(2 * n):
        idx = i % n
        
        while stack and nums[stack[-1]] < nums[idx]:
            result[stack.pop()] = nums[idx]
        
        if i < n:
            stack.append(idx)
    
    return result
```

**4. Largest Rectangle in Histogram**
```python
def largestRectangleArea(heights):
    stack = []
    max_area = 0
    heights.append(0)
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        stack.append(i)
    
    return max_area
```

**5. Trapping Rain Water**
```python
def trap(height):
    if not height:
        return 0
    
    water = 0
    stack = []
    
    for i in range(len(height)):
        while stack and height[stack[-1]] < height[i]:
            bottom = stack.pop()
            
            if not stack:
                break
            
            width = i - stack[-1] - 1
            bounded_height = min(height[i], height[stack[-1]]) - height[bottom]
            water += width * bounded_height
        
        stack.append(i)
    
    return water
```

**6. Remove K Digits**
```python
def removeKdigits(num, k):
    stack = []
    
    for digit in num:
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        
        stack.append(digit)
    
    stack = stack[:-k] if k > 0 else stack
    
    result = ''.join(stack).lstrip('0')
    return result if result else '0'
```

**7. Online Stock Span**
```python
class StockSpanner:
    def __init__(self):
        self.stack = []  # (price, span)
    
    def next(self, price):
        span = 1
        
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        
        self.stack.append((price, span))
        return span
```

**8. Sum of Subarray Minimums**
```python
def sumSubarrayMins(arr):
    MOD = 10**9 + 7
    n = len(arr)
    
    left = [-1] * n
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] > arr[i]:
            stack.pop()
        left[i] = stack[-1] if stack else -1
        stack.append(i)
    
    right = [n] * n
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()
        right[i] = stack[-1] if stack else n
        stack.append(i)
    
    result = 0
    for i in range(n):
        result = (result + arr[i] * (i - left[i]) * (right[i] - i)) % MOD
    
    return result
```

**9. Maximal Rectangle**
```python
def maximalRectangle(matrix):
    if not matrix:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0
    
    for row in matrix:
        for i in range(cols):
            heights[i] = heights[i] + 1 if row[i] == '1' else 0
        
        # Use largest rectangle in histogram
        max_area = max(max_area, largestRectangleArea(heights))
    
    return max_area

def largestRectangleArea(heights):
    stack = []
    max_area = 0
    heights.append(0)
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        stack.append(i)
    
    heights.pop()
    return max_area
```

**10. 132 Pattern**
```python
def find132pattern(nums):
    stack = []
    third = float('-inf')
    
    # Traverse from right to left
    for i in range(len(nums) - 1, -1, -1):
        if nums[i] < third:
            return True
        
        while stack and stack[-1] < nums[i]:
            third = stack.pop()
        
        stack.append(nums[i])
    
    return False
```

### ⚠️ Edge Cases & Pitfalls

- **Empty array**: Handle len(nums) == 0
- **Stack not empty at end**: May need sentinel value
- **Indices vs values**: Store indices for distance calculations
- **Monotonic order**: Choose increasing/decreasing carefully
- **Equal elements**: Decide if pop or not (< vs <=)
- **Circular arrays**: Process twice or use modulo

### ⏱️ Time & Space Complexity

| Problem | Time | Space | Notes |
|---------|------|-------|-------|
| Next greater element | O(n) | O(n) | Each element pushed/popped once |
| Daily temperatures | O(n) | O(n) | Linear scan |
| Largest rectangle | O(n) | O(n) | Each bar processed once |
| Trapping rain water | O(n) | O(n) | Single pass |
| Sum of subarray mins | O(n) | O(n) | Two passes |

### 🧠 Interview Tips

- **Explain monotonic property**: "I'll maintain decreasing stack to find next greater"
- **Why O(n)**: "Each element pushed and popped at most once"
- **Choose type**: "Decreasing stack for next greater, increasing for next smaller"
- **Sentinel value**: "I'll append 0 to flush remaining elements"
- **Distance vs value**: "Store indices to calculate distance"

**Common follow-ups:**
- "What if circular array?" → Process twice or use modulo
- "Can you do it in one pass?" → Yes, with monotonic stack
- "What about previous greater?" → Similar but scan left to right

**Red flags to avoid:**
- Wrong monotonic order (increasing vs decreasing)
- Not handling empty stack
- Forgetting to append current element
- Off-by-one in width calculations

---

## 🎯 Summary: When to Use Each Pattern

| Pattern | Use When | Time | Key Insight |
|---------|----------|------|-------------|
| **Divide & Conquer** | Independent subproblems | O(n log n) | Split, solve, combine |
| **Sweep Line** | Events on timeline | O(n log n) | Process events in order |
| **Interval Problems** | Ranges/overlaps | O(n log n) | Sort by start/end |
| **Monotonic Stack** | Next greater/smaller | O(n) | Maintain order in stack |

---

**🎓 Final Master Tips for Interviews:**

1. **Pattern recognition**: "This looks like [pattern] because..."
2. **Complexity analysis**: Always state time and space
3. **Edge cases**: Empty input, single element, duplicates
4. **Optimization**: "Could optimize from O(n²) to O(n log n) with..."
5. **Trade-offs**: "Using more space for better time complexity"
6. **Alternatives**: "Could also solve with [other approach], but this is better because..."

**Interview flow:**
1. Clarify problem and constraints
2. Discuss approach and pattern
3. Analyze complexity
4. Code solution
5. Test with examples
6. Discuss optimizations

---
