"""
https://leetcode.com/problems/jump-game-v/

Given an array of integers arr and an integer d. In one step you can jump from index i to index:

i + x where: i + x < arr.length and 0 < x <= d.
i - x where: i - x >= 0 and 0 < x <= d.
In addition, you can only jump from index i to index j if arr[i] > arr[j] and arr[i] > arr[k] for all indices k between i and j (More formally min(i, j) < k < max(i, j)).

You can choose any index of the array and start jumping. Return the maximum number of indices you can visit.

Notice that you can not jump outside of the array at any time.
"""


from functools import *
import numpy as np
from typing import *


class Solution:
    def maxJumps(self, nums: List[int], d: int) -> int:
        """
        We can proceed in two phases:
        1. for each point, find the previous and next reachable points
        2. explore the graph that is built from this: since it is a DAG, longest
           paths algorithms are exist and are efficient

        The first step is not that easy:
        - the most efficient way is to build a mono-stack, always decreasing
        - upon adding an entry, pop the smallest elements
        - the ones you pop are reachable
        """

        N = len(nums)

        left_jumps = [[] for _ in range(N)]
        left = [(-1, np.inf)]
        for i in range(N):
            while left[-1][1] < nums[i]:
                j, _ = left.pop()
                if abs(j - i) <= d:
                    left_jumps[i].append(j)
            left.append((i, nums[i]))

        right_jumps = [[] for _ in range(N)]
        right = [(-1, np.inf)]
        for i in reversed(range(N)):
            while right[-1][1] < nums[i]:
                j, _ = right.pop()
                if abs(j - i) <= d:
                    right_jumps[i].append(j)
            right.append((i, nums[i]))

        @lru_cache(maxsize=None)
        def longest_path_from(pos: int):
            left_max = max((longest_path_from(p) for p in left_jumps[pos]), default=0)
            right_max = max((longest_path_from(p) for p in right_jumps[pos]), default=0)
            return max(1 + left_max, 1 + right_max)

        return max(longest_path_from(i) for i in range(N))
