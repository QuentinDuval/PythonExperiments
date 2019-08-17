"""
https://leetcode.com/problems/different-ways-to-add-parentheses/

Given a string of numbers and operators, return all possible results from computing all the different possible ways
to group numbers and operators. The valid operators are +, - and *.
"""

from typing import *


class Solution:
    def diffWaysToCompute(self, expr: str) -> List[int]:
        """
        Parse the expression, then try all different evaluation order by selecting which operation is selected first.
        """

        def backtrack(lo: int, hi: int):
            if lo == hi:
                yield tokens[lo]
                return

            for i in range(lo + 1, hi, 2):
                for left in backtrack(lo, i - 1):
                    for right in backtrack(i + 1, hi):
                        yield tokens[i](left, right)

        tokens = self.parse(expr)
        return list(backtrack(0, len(tokens) - 1))

    def parse(self, expr: str) -> List['token']:
        tokens = []
        i = 0
        while i < len(expr):
            if expr[i].isdigit():
                j = i + 1
                while j < len(expr) and expr[j].isdigit():
                    j += 1
                tokens.append(int(expr[i:j]))
                i = j - 1
            elif expr[i] == '*':
                tokens.append(lambda x, y: x * y)
            elif expr[i] == '-':
                tokens.append(lambda x, y: x - y)
            elif expr[i] == '+':
                tokens.append(lambda x, y: x + y)
            i += 1
        return tokens
