from functools import lru_cache
from typing import List


class Solution:
    def dieSimulator(self, nb_rolls: int, rollMax: List[int]) -> int:
        """
        SOLUTION HERE IS WRONG BECAUSE IT IS "CONSECUTIVE TIME"

        We could try to proceed recursively, roll by roll:
        - roll a dice, recurse with nb_rolls-1, and remove the dice rolled from rollMax
        => leads to too many sub-problems: O(nb_rolls**7)

        Instead, we could try to proceed recursively by die:
        - pick the number of roll you want from the dice from 0 to rollMax[die]
        - recurse to the next die with that many roll left, and only the [die+1,6] remaining
        => leads to O(nb_rolls * 6) sub-problems with O(rollMax[die]) by turn

        The recursion is a bit tricky and must be thought about:
        - if we pick 0 roll from our die => recursive x 1
        - if we pick 1 roll from our die => recursive x rolls_remaining
        - if we pick 2 roll from our die => recursive x rolls_remaining x (rolls_remaining-1) / 2
        => In general: pick k rolls => recursive x (k among rolls_remaining)
        """

        '''
        def binomial(k, n):
            multiplier = 1
            for i in range(k):
                multiplier *= n - i
            denominator = 1
            for i in range(1, k+1):
                denominator *= i
            return multiplier // denominator

        memo = {}

        def visit(die_index: int, nb_roll_remaining: int):
            if nb_roll_remaining == 0:
                memo[(die_index, nb_roll_remaining)] = 1
                return 1
            if die_index >= len(rollMax):
                return 0

            count = memo.get((die_index, nb_roll_remaining))
            if count is not None:
                return count

            count = 0
            max_roll = min(nb_roll_remaining, rollMax[die_index])
            for roll in range(max_roll+1):
                sub = visit(die_index + 1, nb_roll_remaining - roll)
                count += sub * binomial(roll, nb_roll_remaining)
            memo[(die_index, nb_roll_remaining)] = count
            return count

        res = visit(0, nb_rolls) % 1000000007
        return res
        '''

        """
        SOLUTION, THIS TIME WITH 'CONSECUTIVE'
        """

        @lru_cache(maxsize=None)
        def visit(nb_roll_remaining: int, last_played: int):
            if nb_roll_remaining == 0: return 1

            count = 0
            for i in range(6):
                if i != last_played:
                    max_roll = min(nb_roll_remaining, rollMax[i])
                    for roll in range(1, max_roll + 1):
                        count += visit(nb_roll_remaining - roll, i)
            return count % 1000000007

        return visit(nb_rolls, -1) % 1000000007
