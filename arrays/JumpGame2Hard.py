"""
https://leetcode.com/problems/jump-game-ii

Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Your goal is to reach the last index in the minimum number of jumps.
"""

from collections import *
from typing import List


class Solution:

    def jump_1(self, nums: List[int]) -> int:
        """
        Solution 1:
        Simply try every possibility !

        You have to memoize the sub-problems for there will be overlaps
        - Number of sub-problems: O(N)
        - Complexity of recursion: at the very most O(N)
        => O(N**2) time complexity and O(N) space complexity

        Timeouts on inputs such as [N, N-1, N-2, ...] and N = 25000
        Need a better algorithm than O(N**2)
        """

        n = len(nums)
        memo = [float('inf')] * n
        memo[-1] = 0
        for i in reversed(range(n-1)):
            max_jump = nums[i]
            if max_jump >= n-1-i:
                memo[i] = 1
            else:
                sub_problems = (memo[i+jump] for jump in range(1, max_jump+1))
                memo[i] = 1 + min(sub_problems, default=float('inf'))

        return memo[0]

    def jump_2(self, nums: List[int]) -> int:
        """
        Solution 2:
        You can do much simpler actually, by seeing this is just a SHORTEST PATH.
        => You do not even need Dijsktra, just a BFS

        Complexity:
        - Space: O(N) maximum (if you use 'discovered' set)
        - Time: O(N**2) as all edges can be visited

        This passes the tests, only if we explore the solutions in the right order
        => Need to try the longest jumps first (avoid hitting N**2 edges visited)

        Time: 64ms (36%)
        """

        if len(nums) <= 1:
            return 0

        to_visit = deque()
        to_visit.append((0, 0))
        discovered = {0}

        while to_visit:
            position, step = to_visit.popleft()
            max_jump = nums[position]
            for jump in reversed(range(1, max_jump+1)): # 'reverse' is critical
                destination = position + jump
                if destination >= len(nums) - 1:
                    return step + 1
                if destination not in discovered:
                    discovered.add(destination)
                    to_visit.append((destination, step+1))

        return float('inf')

    def jump_3(self, nums: List[int]) -> int:
        """
        Solution 3: Greedy !

        Let's say we are at K jumps:
        - the range it can reach is [0, end_range]:
        - 'farthest' is the farthest point that points in [0, end_range] can reach
        - once the current point reaches end_range:
          * then trigger another jump
          * set the new end_range as farthest
          * repeat

        It works because:
        1. we keep expanding a circle of places reachable with increasing K steps
        2. if we can jump K, we can jump K-1 (so all before are reachable)

        It is much like a kind of BFS, but more memory efficient.
        - Time complexity: O(N)
        - Space complexity: O(1)

        Time: 44ms (98%)
        """

        jump_nb = 0
        end_range = 0
        farthest = 0
        for i, n in enumerate(nums[:-1]):
            if i + n > farthest:
                farthest = i + n
            if farthest >= len(nums) - 1:
                return jump_nb + 1
            if i == end_range:
                jump_nb += 1
                end_range = farthest
        return jump_nb

