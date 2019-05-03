"""
https://practice.geeksforgeeks.org/problems/maximum-index/0

Given an array A[] of N positive integers.
The task is to find the maximum of j - i subjected to the constraint of A[i] <= A[j].
"""


"""
How to approach it?

The brute force approach is to try for every i < j, but:
- It is useless to try at every i: we try only at a successive index 'i' if its value is LOWER than all BEFORE
- It is useless to try at every j: we try only at a preceding index 'j' if its value is HIGHER than all AFTER
=> Note there is a kind of symmetry.

Still, even if we only try these combinations, the complexity is O(N**2):
- Example of the collection [5, 10, 4, 9, 3, 8, 2, 7, 1, 6]

SOLUTION:
- We keep an ordered map of values collected from right to left (we only add values higher than all previous) to index
- We scan from left to right, and try each value (search lower_bound) and compute the difference of index

COMPLEXITY:
- O(N log N) time
- O(N) space

EXAMPLE:
[34, 8, 10, 3, 2, 80, 30, 33, 1]
- map contains [1 => 8, 33 => 7, 80 => 5] (this is just a vector & we can binary search in it)
- then we look for 34 at index 0 => we get 80 at index 5 => 5
- then we look for 8 at index 1 => we get 33 at index 7 => 6
- and so on...
- BUT we need to drop the head of the stack each time we move pass through an index
"""


def maximum_interval(nums):
    n = len(nums)
    right_bigs = [(nums[-1], n - 1)]

    def search_next_big(val):
        lo = 0
        hi = len(right_bigs) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if val <= right_bigs[mid][0]:   # <= is critical to get the lower_bound !
                hi = mid - 1
            else:
                lo = mid + 1
        return lo

    for i in reversed(range(n - 1)):
        if nums[i] > right_bigs[-1][0]:
            right_bigs.append((nums[i], i))

    max_diff = 0
    for i, val in enumerate(nums):
        if i == right_bigs[-1][1]:
            right_bigs.pop()
        else:
            pos = search_next_big(val)
            if pos < len(right_bigs):
                max_diff = max(max_diff, right_bigs[pos][1] - i)

    return max_diff


print(maximum_interval([34, 8, 10, 3, 2, 80, 30, 33, 1]))  # Expect 6
print(maximum_interval([4, 3, 2, 1]))  # Expect 0
print(maximum_interval([1, 2, 3, 4]))  # Expect 3

