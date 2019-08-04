"""
https://leetcode.com/problems/pancake-sorting/

Given an array A, we can perform a pancake flip:
We choose some positive integer k <= A.length, then reverse the order of the first k elements of A.
We want to perform zero or more pancake flips (doing them one after another in succession) to sort the array A.

Return the k-values corresponding to a sequence of pancake flips that sort A.
Any valid answer that sorts the array within 10 * A.length flips will be judged as correct.
"""

from typing import List


class Solution:
    def pancakeSort(self, nums: List[int]) -> List[int]:
        """
        One basic strategy consist in moving the bigger element at the end:
        - this can be done in 2 moves (put it in front, put it in the end)
        - and then recurse with a smaller table

        The bigger element is necessarily len(nums) and can be found in O(N)
        The cost of swapping is O(N)
        We have to do it O(N) times
        => O(N ** 2) algorithm in swaps, O(N) in flips
        """

        flips = []

        def flip(count):
            flips.append(count)
            nums[:count] = nums[:count][::-1]

        def find_max(up_to):
            val = pos = 0
            for i in range(up_to + 1):
                if nums[i] > val:
                    val = nums[i]
                    pos = i
            return val, pos

        # The recurrence can be expressed in index of end of array
        for hi in reversed(range(1, len(nums))):
            val, pos = find_max(up_to=hi)
            if pos > 0:
                flip(pos + 1)  # Do a flip to bring the max value at front
            flip(hi + 1)  # Do a flip to bring the max value at the end

        return flips
