"""
https://leetcode.com/problems/n-queens
"""

from typing import List


# TODO - show how easy it is to create a lazy stream in Python
# TODO - show that if you want more 'iterator' like capacity (like reverse), just put it inside a list


class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        """
        We have to place queue on each row: we only need to find the column.

        Back-track to find all positions possible by row, with pruning.
        Return all possible solutions.
        """

        def build_solution(queens):
            return ["." * y + "Q" + "." * (n - y - 1) for y in queens]

        def is_last_valid(queens):
            last_x = len(queens) - 1
            last_y = queens[-1]
            for x, y in enumerate(queens[:-1]):
                if abs(last_x - x) == abs(last_y - y):
                    return False
            return True

        def backtrack(queens):
            # In 'queens', value 'j' at index 'i' represent position (i, j)
            if len(queens) == n:
                yield build_solution(queens)
                return

            for col in range(n):
                if col not in queens:
                    queens.append(col)
                    if is_last_valid(queens):
                        yield from backtrack(queens)    # Do not forget the 'yield from' or else it has no effect
                    queens.pop()

        return list(backtrack([]))
