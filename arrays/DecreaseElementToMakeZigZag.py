"""
https://leetcode.com/problems/decrease-elements-to-make-array-zigzag/

Given an array nums of integers, a move consists of choosing any element and decreasing it by 1.

An array A is a zigzag array if either:

Every even-indexed element is greater than adjacent elements, ie. A[0] > A[1] < A[2] > A[3] < A[4] > ...
OR, every odd-indexed element is greater than adjacent elements, ie. A[0] < A[1] > A[2] < A[3] > A[4] < ...

Return the minimum number of moves to transform the given array nums into a zigzag array.
"""

from typing import List


class Solution:
    def movesToMakeZigzag(self, nums: List[int]) -> int:
        """
        There is not difference between moving previous up or decreasing current.
        Ex: [9,6,1,6,2] gives [9,10,1,6,2] or [5,6,1,6,2] with same cost

        If you increase something, you make the decrease of next one all the simplest.
        So, the algorithm can be simplified to: adjust violated numbers as they go.

        We just need to try both starting with decrease and starting with increase.
        => Two passes, for a O(N) algorithm

        But the problem statement says DECREASE BY 1 (not increase)
        """

        if not nums:
            return 0
        return min(self.movesToZigZag(nums, True), self.movesToZigZag(nums, False))

    def movesToZigZag(self, nums: List[int], start_by_decrease: bool) -> int:
        cost = 0
        decrease = start_by_decrease
        prev = nums[0]
        for curr in nums[1:]:
            if decrease and curr >= prev:
                cost += curr - (prev - 1)  # curr go down to prev-1
                prev = prev - 1
            elif not decrease and curr <= prev:
                cost += prev - (curr - 1)  # prev goes further down to curr - 1
                prev = curr
            else:
                prev = curr
            decrease = not decrease
        return cost
