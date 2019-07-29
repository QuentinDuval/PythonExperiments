"""
https://leetcode.com/problems/partition-array-for-maximum-sum/

Given an integer array A, you partition the array into (contiguous) subarrays of length at most K.
After partitioning, each subarray has their values changed to become the maximum value of that subarray.

Return the largest sum of the given array after partitioning.
"""

from functools import lru_cache
from typing import List


class Solution:
    def maxSumAfterPartitioning(self, nums: List[int], k: int) -> int:
        """
        Each decision depends on the previous ones (and impacts on the rest)
        => There is not greedy solution visible like this

        We have to try everything. To do so, we try every possible slices from
        left to right (or right to left - just pick a direction) and recurse.

        There are obviously overlapping sub-problems here, so we need to use DP.
        - Number of sub-problems: O(n)
        - Cost by sub-problem: O(k)
        => Time complexity is O(n * k)
        => Space complexity is O(n), but can be optimized to O(k) doing it right
        """

        @lru_cache(maxsize=None)
        def maximize(pos: int) -> int:
            if pos + k >= len(nums):
                return (len(nums) - pos) * max(nums[pos:], default=0)

            max_val = float('-inf')
            slice_max = float('-inf')
            for i in range(0, k):
                slice_max = max(slice_max, nums[pos + i])
                max_val = max(max_val, (i + 1) * slice_max + maximize(pos + i + 1))
            return max_val

        return maximize(0)
