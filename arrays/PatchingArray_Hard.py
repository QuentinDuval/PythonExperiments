"""
https://leetcode.com/problems/patching-array/

Given a sorted positive integer array nums and an integer n, add/patch elements to the array such that any number
in range [1, n] inclusive can be formed by the sum of some elements in the array.

Return the minimum number of patches required.
"""

from typing import List


class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        """
        The idea is simple (and then we can optimize it interestingly)

        Core idea:
        - you have to try all numbers from 1 to n
        - you need to track which numbers are "reachable"

        First idea:
        - sort the nums in input
        - add them one by one:
            - for each one, complete the list of possible numbers (a hash set)
            - if one number 'i' < n cannot be constructed, add i (and complete the list)

        Example:
        [1, 5, 10], n = 20
        - [1] {1} you cannot find 2, try to consume 5, but 5 is too high, so add 2
        - [1,2] {1, 2, 3} you cannot find 4, try to consume 5, but 5 is too high, so add 4
        - [1, 2, 4] {1, 2, 3, 4, 5, 6, 7}, you cannot find 8, consume 5 (low enough)
        - [1, 2, 4, 5] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} ... etc ...

        But then you can see that the hash set is only containing contiguous ranges, so you
        can just play with a 'end_of_range' cursor (and move it until it reaches past 'n').

        Beat 97%
        """

        nums.sort()
        patches = []

        pos = 0  # Next number to consume in provided list
        max_range = 0  # Current end of range
        while max_range < n:

            # Consume from the provided list if possible
            if pos < len(nums) and nums[pos] <= max_range + 1:
                max_range += nums[pos]
                pos += 1

            # Else add a new number to the list (the patches)
            else:
                patches.append(max_range + 1)
                max_range += patches[-1]

        return len(patches)



