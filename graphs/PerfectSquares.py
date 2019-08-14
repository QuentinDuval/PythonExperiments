"""
https://leetcode.com/problems/perfect-squares/

Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.
"""

from collections import deque
import itertools


class Solution:
    def numSquares(self, n: int) -> int:
        """
        The greedy approach does not work:
        - taking the maximum square that fits might increase the number of perfect square needed
        - example of 12 = 4 + 4 + 4 and not 9 + 1 + 1 + 1

        We can do it recursively with dynamic programming:
        - numSquares(n-i*i) where i is the selected square
        - it would explore O(N) sub-problems (because 1 is a perfect square)
        => O(N * sqrt(N)) total complexity
        => But this does not pass the test cases (it explores too much)

        The thing is that we do not need to explore all solutions: we can cut branches.
        If we already found a solution with K squares, then no need to explore after K depth.

        The BFS does automatically this.
        """

        if n == 0:
            return 0

        # Construct the list of perfect squares (improves performance)
        # Reversing it would be better but searching for first would require binary search
        squares = []
        for i in range(1, n + 1):
            if i * i > n:
                break
            squares.append(i * i)

        # BFS, looking for zero
        depth = 0
        discovered = {n}
        to_visit = {n}
        while to_visit:
            next_to_visit = set()
            for val in to_visit:
                for square in squares:
                    next_val = val - square
                    if next_val < 0:
                        break
                    if next_val == 0:
                        return 1 + depth
                    if next_val not in discovered:
                        discovered.add(next_val)
                        next_to_visit.add(next_val)
            to_visit = next_to_visit
            depth += 1

