"""
https://leetcode.com/problems/predict-the-winner/

Given an array of scores that are non-negative integers.
Player 1 picks one of the numbers from either end of the array followed by the player 2 and then player 1 and so on.
Each time a player picks a number, that number will not be available for the next player.
This continues until all the scores have been chosen.
The player with the maximum score wins.

Given an array of scores, predict whether player 1 is the winner.
You can assume each player plays to maximize his score.
"""

from typing import List


def cache(f):
    memo = {}

    def wrapper(*args):
        res = memo.get(args)
        if res is not None:
            return res
        res = f(*args)
        memo[args] = res
        return res

    return wrapper


class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        """
        Return the best score advantage you can get from:
        - Playing one valid move
        - Plug the negation of the best score advantage of the opponent (since it is a negative sum game)

        Complexity is O(N**2) for this is the number of sub-solutions (i < j)
        """

        # Top down solution
        if not nums:
            return None

        @cache
        def bestScoreDiff(lo, hi):
            if lo > hi:
                return 0
            return max(
                nums[lo] - bestScoreDiff(lo+1, hi),
                nums[hi] - bestScoreDiff(lo, hi-1))

        return bestScoreDiff(0, len(nums)-1) >= 0

    def PredictTheWinner2(self, nums: List[int]) -> bool:
        """
        Bottom-up solution, that reduces space to O(N)

        The key is to draw the game tree and see that left-right is the same as right-left:

            *
           / \
          *   *
         / \ / \
        *   *   *

        So we can start from the lower-stage and work our way up.
        We can also see that the size of the game-tree is N(N+1) / 2
        """

        if not nums:
            return None

        # Memo[i] represent the best solution for [i, i+k] at step k
        memo = list(nums)

        for span in range(1, len(nums)):
            new_memo = []
            for i in range(len(nums) - span):
                new_memo.append(max(nums[i] - memo[i + 1], nums[i + span] - memo[i]))
            memo = new_memo

        return memo[0] >= 0

