"""
https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/

Given an array of integers sorted in ascending order, find the starting and ending position of a given target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return [-1, -1].
"""

from typing import List


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        Beats 98%
        """

        def lower_bound():
            lo = 0
            hi = len(nums) - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if nums[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return lo

        def upper_bound():
            lo = 0
            hi = len(nums) - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if nums[mid] <= target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return lo

        lo = lower_bound()
        hi = upper_bound()
        if lo == hi:
            return [-1, -1]
        else:
            return [lo, hi - 1]
