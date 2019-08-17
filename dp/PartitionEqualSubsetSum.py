"""
https://leetcode.com/problems/partition-equal-subset-sum/

Given a non-empty array containing only positive integers, find if the array can be partitioned into TWO subsets such
that the sum of elements in both subsets is equal.

Note:
* Each of the array element will not exceed 100.
* The array size will not exceed 200.
"""

from functools import lru_cache
from typing import List


class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2 == 1:
            return False

        half = total // 2

        @lru_cache(maxsize=None)
        def search(pos: int, remaining: int) -> bool:
            if remaining < 0:
                return False
            if remaining == 0:
                return True
            if pos == len(nums):
                return False
            return search(pos + 1, remaining - nums[pos]) or search(pos + 1, remaining)

        return search(0, half)
