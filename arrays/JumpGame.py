"""
https://leetcode.com/problems/jump-game/

Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.
"""


from functools import *
from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        """
        Memoization based solution in O(N**2)
        Timeout
        """

        if len(nums) <= 1:
            return True

        '''
        n = len(nums)
        memo = [False] * n
        memo[-1] = True

        zero_found = False
        for i in reversed(range(n-1)):
            if nums[i] == 0:
                zero_found = True
                memo[i] = False
            elif not zero_found:
                memo[i] = True
            elif i+nums[i] >= n-1:
                memo[i] = True
            else:    
                for jump in range(1, nums[i]+1):
                    if memo[i+jump]:
                        memo[i] = True
        return memo[0]
        '''

        """
        The better idea is to play in terms of intervals:
        - if we can jump N, we can jump N-1
        - so just extend the limit while you can
        """

        i = 0
        reachable = 0
        while i <= reachable:
            reachable = max(reachable, i + nums[i])
            if reachable >= len(nums) - 1:
                return True
            i += 1
        return False
