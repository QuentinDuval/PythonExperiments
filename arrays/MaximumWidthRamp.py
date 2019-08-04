"""
https://leetcode.com/problems/maximum-width-ramp/

Given an array A of integers, a ramp is a tuple (i, j) for which i < j and A[i] <= A[j].
The width of such a ramp is j - i.

Find the maximum width of a ramp in A.  If one doesn't exist, return 0.
"""


from typing import List


class Solution:
    def maxWidthRamp(self, nums: List[int]) -> int:
        """
        The basic idea is to scan the inputs from the left and look
        whether or not we found some number that is lower on the left.

        The problem is that we want to take:
        - the lowest number that is furthest on the left.
        - not necessarily the lowest number overall

        What we could do is use a BST:
        - when we add a new number, only add it if it is a new minimum
        - then search for the number below the current one (it is garanteed to be the furthest one)
        => A simple array would do (sorted in the descending order)

        Time complexity is O(N log N)
        Beats only 6%
        """

        # TODO - there is a two pass left to right, right to left algorithm that does it in O(N)

        if len(nums) <= 1:
            return 0

        max_ramp_width = 0
        prev_idx = [0]
        prev_val = [nums[0]]
        for i in range(1, len(nums)):
            num = nums[i]
            if num < prev_val[-1]:
                prev_idx.append(i)
                prev_val.append(num)
            else:
                j = self.inverted_lower_bound(prev_val, num)
                max_ramp_width = max(max_ramp_width, i - prev_idx[j])
        return max_ramp_width

    def inverted_lower_bound(self, vals, val):
        lo = 0
        hi = len(vals) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if vals[mid] == val:
                return mid
            elif vals[mid] >= val:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo  # First value lower or equal than val (hi is first value higher than val)
