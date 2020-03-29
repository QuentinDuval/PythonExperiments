"""
https://leetcode.com/problems/parsing-a-boolean-expression/

Return the result of evaluating a given boolean expression, represented as a string.

An expression can either be:

"t", evaluating to True;
"f", evaluating to False;
"!(expr)", evaluating to the logical NOT of the inner expression expr;
"&(expr1,expr2,...)", evaluating to the logical AND of 2 or more inner expressions expr1, expr2, ...;
"|(expr1,expr2,...)", evaluating to the logical OR of 2 or more inner expressions expr1, expr2, ...
"""


from typing import *


class Solution:
    def parseBoolExpr(self, expression: str) -> bool:

        def parse(i: int) -> Tuple[bool, int]:

            def parse_args(i: int) -> Tuple[List[bool], int]:
                assert expression[i] == '('
                bs = []
                b, i = parse(i + 1)
                bs.append(b)
                while expression[i] == ',':
                    b, i = parse(i + 1)
                    bs.append(b)
                assert expression[i] == ')'
                return bs, i + 1

            if expression[i] == 't':
                return True, i + 1

            if expression[i] == 'f':
                return False, i + 1

            if expression[i] == '!':
                assert expression[i + 1] == '('
                b, i = parse(i + 2)
                assert expression[i] == ')'
                return not b, i + 1

            if expression[i] == '&':
                bs, i = parse_args(i + 1)
                return all(bs), i

            if expression[i] == '|':
                bs, i = parse_args(i + 1)
                return any(bs), i

        b, i = parse(0)
        return b
