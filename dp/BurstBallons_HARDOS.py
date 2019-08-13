"""
https://leetcode.com/problems/burst-balloons/

Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number on it represented by array nums.
You are asked to burst all the balloons. If the you burst balloon i you will get nums[left] * nums[i] * nums[right] coins.
Here left and right are adjacent indices of i. After the burst, the left and right then becomes adjacent.

Find the maximum coins you can collect by bursting the balloons wisely.

Note:
* You may imagine nums[-1] = nums[n] = 1. They are not real therefore you can not burst them.
* 0 ≤ n ≤ 500, 0 ≤ nums[i] ≤ 100
"""

from functools import lru_cache
from typing import List


class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        """
        At each iteration, select the ballon you will pop last (and will pop with the borders) and
        then solve recursively each havles.

        Other recursion strategy (like choosing the balloon you pop first) makes memoization harder
        and provoke a combinatorial explosion.

        Complexity is O(N ** 3)
        """

        nums = [1] + nums + [1]

        @lru_cache(maxsize=None)
        def dp(l_border: int, r_border: int) -> int:
            if l_border >= r_border:
                return 0

            max_score = 0
            for i in range(l_border + 1, r_border):
                score = nums[l_border] * nums[i] * nums[r_border]
                max_score = max(max_score, score + dp(l_border, i) + dp(i, r_border))
            return max_score

        return dp(0, len(nums) - 1)
