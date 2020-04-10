"""
https://leetcode.com/problems/stone-game-ii

Alex and Lee continue their games with piles of stones.  There are a number of piles arranged in a row,
and each pile has a positive integer number of stones piles[i].  The objective of the game is to end with the most stones.

Alex and Lee take turns, with Alex starting first.  Initially, M = 1.

On each player's turn, that player can take all the stones in the first X remaining piles, where 1 <= X <= 2M.
Then, we set M = max(M, X).

The game continues until all the stones have been taken.

Assuming Alex and Lee play optimally, return the maximum number of stones Alex can get.
"""

from functools import lru_cache
from typing import List


class Solution:
    def stoneGameII(self, piles: List[int]) -> int:
        N = len(piles)

        @lru_cache(maxsize=None)
        def visit(i: int, m: int, is_player: bool) -> int:
            if i >= N:
                return 0

            best_score = -1
            for x in range(1, 2 * m + 1):
                if i + x > N:
                    break

                sub_score = visit(i + x, max(m, x), not is_player)
                if is_player:
                    sub_score += sum(piles[i:i + x])
                    best_score = max(best_score, sub_score)
                elif best_score == -1:
                    best_score = sub_score
                else:
                    best_score = min(best_score, sub_score)
            return best_score

        return visit(0, 1, True)
