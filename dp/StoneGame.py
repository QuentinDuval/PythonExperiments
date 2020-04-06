"""
https://leetcode.com/problems/stone-game/

Alex and Lee play a game with piles of stones.  There are an even number of piles arranged in a row,
and each pile has a positive integer number of stones piles[i].

The objective of the game is to end with the most stones.  The total number of stones is odd, so there are no ties.

Alex and Lee take turns, with Alex starting first.  Each turn, a player takes the entire pile of stones from either the
beginning or the end of the row.  This continues until there are no more piles left, at which point the person with the
most stones wins.

Assuming Alex and Lee play optimally, return True if and only if Alex wins the game.
"""


from functools import lru_cache
import numpy as np
from typing import List


class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        """
        Number of sub-problems: O(N**2)
        Complexity of sub-problem: O(1)
        => O(N**2) total complexity
        """

        '''
        @lru_cache(maxsize=None)
        def best_score_from(lo: int, hi: int) -> int:
            if lo == hi:
                return piles[lo]

            l_op_score = piles[lo] - best_score_from(lo+1, hi)
            r_op_score = piles[hi] - best_score_from(lo, hi-1)
            return max(l_op_score, r_op_score)

        alex_score = best_score_from(0, len(piles)-1)
        return alex_score > 0
        '''

        """
        Top-bottom recursion with O(N**2) storage
        """

        '''
        N = len(piles)
        scores = np.diag(piles)
        for lo in reversed(range(N)):
            for hi in range(lo+1, N):
                l_op_score = piles[hi] - scores[lo,hi-1]
                r_op_score = piles[lo] - scores[lo+1,hi]
                scores[lo,hi] = max(l_op_score, r_op_score)
        return scores[0,N-1]
        '''

        """
        Top-bottom recursion with O(N) storage
        """

        N = len(piles)
        scores = [0] * N
        for lo in reversed(range(N)):
            new_scores = [0] * N
            new_scores[lo] = piles[lo]
            for hi in range(lo + 1, N):
                l_op_score = piles[hi] - new_scores[hi - 1]
                r_op_score = piles[lo] - scores[hi]
                new_scores[hi] = max(l_op_score, r_op_score)
            scores = new_scores
        return scores[0]
