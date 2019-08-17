"""
https://leetcode.com/problems/generate-parentheses/

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
"""


from typing import List


class Solution:
    def generateParenthesis_1(self, n: int) -> List[str]:
        """
        Link with Catalan numbers: 1, 2, 5, 14,..
        C(n) = sum(i = 0 .. n-1) of C(i)(n-1-i)

        Equal to number of root n-ary tree with n+1 nodes
        - '(' goes down
        - ')' goes back up

        Recursion would go like this:
        - choose the size of the first tree
        - then recurse to the left and right

        44ms, beats 49%
        """

        def generate(n):
            if n == 0:
                yield ""
                return

            for i in range(1, n + 1):
                for left in generate(i - 1):
                    for right in generate(n - i):
                        yield '(' + left + ')' + right

        return list(generate(n))

    def generateParenthesis(self, n: int) -> List[str]:
        """
        Other solution is reformulate:
        - try to add '(' while you still have some remaining
        - try to add ')' if some parentheses must be closed

        40 ms, beats 78%
        """

        def generate(curr, opened, remaining):
            if remaining == 0:
                yield curr + opened * ')'
                return

            yield from generate(curr + "(", opened + 1, remaining - 1)
            if opened > 0:
                yield from generate(curr + ")", opened - 1, remaining)

        return list(generate("", 0, n))
