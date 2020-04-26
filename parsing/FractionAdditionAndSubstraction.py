"""
https://leetcode.com/problems/fraction-addition-and-subtraction/

Given a string representing an expression of fraction addition and subtraction, you need to return
the calculation result in string format. The final result should be irreducible fraction.

If your final result is an integer, say 2, you need to change it to the format of fraction that has denominator 1.
So in this case, 2 should be converted to 2/1.
"""


from math import *
from typing import *


class Solution:
    def fractionAddition(self, expr: str) -> str:

        def parse(expr: str) -> Tuple[Tuple[int, int], int]:
            if expr[0] != '-' and expr[0] != '+':
                expr = '+' + expr

            i = 0
            while i < len(expr):
                negative = -1 if expr[i] == '-' else 1
                i += 1
                num = 0
                denom = 0
                while expr[i] != '/':
                    num = num * 10 + int(expr[i])
                    i += 1
                i += 1
                while i < len(expr) and expr[i] != '+' and expr[i] != '-':
                    denom = denom * 10 + int(expr[i])
                    i += 1
                yield (negative * num, denom)

        num, denom = 0, 1
        for a, b in parse(expr):
            p = denom * b // gcd(denom, b)
            num = num * (p // denom) + a * (p // b)
            denom = p

        if num == 0:
            return "0/1"
        else:
            g = gcd(num, denom)
            return str(num // g) + "/" + str(denom // g)
