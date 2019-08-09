"""
https://leetcode.com/problems/first-missing-positive/

Given an unsorted integer array, find the smallest missing positive integer.
"""


from typing import List


class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        """
        The idea is to scan through the table and:
        - skip the element if it is at its correct position
        - swap the element with its correct position if indexes allows (else make sure to modify it to negative)
        - then do a second pass to identify whatever negative elements remain

        Subtelty:
        It is possible to stay in an infinite loop is the swap does not change the table (same value: [1, 1])

        Complexity:
        O(N) because at each step we make progress (move one number to its correct position)
        """

        i = 0
        while i < len(nums):
            # Index is out of bounds - make the position negative and skip it
            if nums[i] > len(nums):
                nums[i] *= -1
                i += 1
                continue

            # Index is negative - skip it
            actual = nums[i]
            expected = i + 1
            if actual <= 0 or actual == expected:
                i += 1
                continue

            # Swap with the correct position if it makes progress
            if nums[actual - 1] != nums[expected - 1]:
                nums[actual - 1], nums[expected - 1] = nums[expected - 1], nums[actual - 1]
            else:
                nums[expected - 1] *= -1
                i += 1

        for i in range(len(nums)):
            if nums[i] <= 0:
                return i + 1
        return len(nums) + 1

