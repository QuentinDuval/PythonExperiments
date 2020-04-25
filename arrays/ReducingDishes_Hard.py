"""
https://leetcode.com/problems/reducing-dishes/

A chef has collected data on the satisfaction level of his n dishes. Chef can cook any dish in 1 unit of time.

Like-time coefficient of a dish is defined as the time taken to cook that dish including previous dishes multiplied by its satisfaction level  i.e.  time[i]*satisfaction[i]

Return the maximum sum of Like-time coefficient that the chef can obtain after dishes preparation.

Dishes can be prepared in any order and the chef can discard some dishes to get this maximum value.
"""
from typing import List


class Solution:
    def maxSatisfaction(self, satisfactions: List[int]) -> int:
        """
        It seems logical to keep the most satisfying dish to the end since the
        time multiplier will be the highest.

        The question is therefore: where to start in the negative?
        """

        satisfactions.sort()

        def score_from(start: int):
            total = 0
            for t, sat in enumerate(satisfactions[start:]):
                total += (t + 1) * sat
            return total

        highest_score = 0
        for start in range(len(satisfactions)):
            highest_score = max(highest_score, score_from(start))
        return highest_score
