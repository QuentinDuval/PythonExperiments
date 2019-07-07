"""
https://leetcode.com/problems/24-game

You have 4 cards each containing a number from 1 to 9.
You need to judge whether they could operated through *, /, +, -, (, ) to get the value of 24.
"""


from typing import *


class Solution:
    def judgePoint24(self, nums: List[int]) -> bool:
        """
        !!! WRONG PROBLEM !!!
        Look at the problem, the order of the card can be changed.

        Do not trouble yourself too much with the parentheses:
        - They are only artifacts of the evaluation of the expression
        - Just build an evaluation tree (implicit is okay) and parenthesisation is selected

        So the goal is to iterate through the intersection between each number:
        - Recur on both sides to see all possible evaluations
        - Try with every operators possible
        - Result all possible evaluation results
        Then at the upper level, search 24.

        BEWARE of overlapping problems:
        - Dynamic programming is needed
        - Number of sub-problems: O(N**2)

        Complexity is exponential but the depth is very limited (4 numbers maximum)
        """

        '''
        from functools import lru_cache
        
        def expr_for(op, lhs, rhs):
            return "(" + lhs + ")" + op + "(" + rhs + ")"

        @lru_cache(maxsize=None) # beware: do not cache a generator as it will be depleted
        def all_possible(lo: int, hi: int) -> List[int]:
            if lo == hi:
                return [(nums[lo], str(nums[lo]))]

            result = []
            for mid in range(lo, hi):
                for lhs, lhs_e in all_possible(lo, mid):
                    for rhs, rhs_e in all_possible(mid+1, hi):
                        result.append((lhs + rhs, expr_for("+", lhs_e, rhs_e)))
                        result.append((lhs - rhs, expr_for("-", lhs_e, rhs_e)))
                        result.append((lhs * rhs, expr_for("*", lhs_e, rhs_e)))
                        if rhs != 0:
                            result.append((lhs // rhs, expr_for("/", lhs_e, rhs_e)))
            return result

        return any(val == 24 for val, expr in all_possible(0, len(nums) - 1))
        '''

        """
        Solve the right problem:
        - Pick two numbers
        - Pick an operation
        - Fusion these numbers
        - Recurse
        """

        def take_2(cards: List[int]):
            for i in range(len(cards)):
                for j in range(i + 1, len(cards)):
                    sub_prob = cards[:i] + cards[i + 1:j] + cards[j + 1:]
                    yield cards[i], cards[j], sub_prob

        def recur(val: int, cards: List[int]):
            cards.append(val)
            yield from visit(cards)
            cards.pop()

        def visit(cards: List[int]):
            if len(cards) == 1:
                yield cards[0]
                return

            for x, y, rest in take_2(cards):
                yield from recur(x + y, rest)
                yield from recur(x - y, rest)
                yield from recur(y - x, rest)
                yield from recur(x * y, rest)
                if y != 0:
                    yield from recur(x / y, rest)
                if x != 0:
                    yield from recur(y / x, rest)

        return any(abs(val - 24) <= 0.00001 for val in visit(nums))
