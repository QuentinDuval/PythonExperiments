"""
https://leetcode.com/problems/longest-increasing-subsequence/
"""

from typing import List


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """
        Beats 96% (44 ms)
        """
        longest = []
        for num in nums:
            i = self.lower_bound(longest, num)
            if i < len(longest):
                longest[i] = num
            else:
                longest.append(num)
        return len(longest)

    def lower_bound(self, nums, searched):
        lo = 0
        hi = len(nums) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] < searched:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo
