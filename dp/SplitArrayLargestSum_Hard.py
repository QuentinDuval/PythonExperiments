"""
https://leetcode.com/problems/split-array-largest-sum/

Given an array which consists of non-negative integers and an integer m, you can split the array into
m non-empty continuous subarrays. Write an algorithm to minimize the largest sum among these m subarrays.

Note:
If n is the length of array, assume the following constraints are satisfied:
* 1 ≤ n ≤ 1000
* 1 ≤ m ≤ min(50, n)
"""


from typing import List


class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        """
        A bit like bin-packing => requires exhaustive search

        Then this is basic dynamic programming, based on simple recursion.
        Number of subproblems is O(m * n) and cost of recursion is O(n)
        => Complexity is O(m * n ^ 2)

        Simple optimizations based on early cutting work as well.
        Beats only 14%
        """

        def cache(f):
            memo = {}

            def wrapped(*args):
                if args in memo:
                    return memo[args]
                res = f(*args)
                memo[args] = res
                return res

            return wrapped

        @cache
        def minimize(partitions: int, pos: int) -> int:
            if pos == len(nums):
                return 0

            if partitions == 1:
                return sum(nums[pos:])

            lowest = float('inf')
            curr_sum = 0
            for i in range(pos, len(nums)):
                curr_sum += nums[i]
                sub = minimize(partitions - 1, i + 1)
                lowest = min(lowest, max(curr_sum, sub))
                if curr_sum > sub:
                    break
            return lowest

        return minimize(m, 0)
