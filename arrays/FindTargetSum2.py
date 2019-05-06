"""
Form with TWO NUMBERS:

Given an array A of N elements.
Complete the function which returns true if a pair exists in array A whose sum is target else return false.
"""

from typing import List


def find_pair_sum(nums: List[int], target: int) -> bool:
    """
    Solution based on sorting: O(N log N)
    """
    nums.sort()

    lo = 0
    hi = len(nums) - 1
    while lo < hi:
        total = nums[lo] + nums[hi]
        if total == target:
            return True
        elif total < target:
            lo += 1
        else:
            hi -= 1

    return False


def find_pair_sum(nums: List[int], target: int) -> bool:
    """
    Solution based on a hash set:
    - Move an index 'i' from left to right
    - Store all previously seen elements
    - Search for target - nums[i]

    Complexity is O(N) time and space
    """
    if len(nums) < 2:
        return False

    visited = set()
    for val in nums:
        if target - val in visited:
            return True
        visited.add(val)
    return False
