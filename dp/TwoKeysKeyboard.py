"""
https://leetcode.com/problems/2-keys-keyboard

Initially on a notepad only one character 'A' is present. You can perform two operations on this notepad for each step:

Copy All: You can copy all the characters present on the notepad (partial copy is not allowed).
Paste: You can paste the characters which are copied last time.


Given a number n. You have to get exactly n 'A' on the notepad by performing the minimum number of steps permitted.
Output the minimum number of steps to get n 'A'.
"""


class Solution:
    def minSteps(self, n: int) -> int:
        """
        Sub problem?
        - you have a number of A available at start (must copy them)
        - you have a remaining number of steps
        Overlapping sub-problems? Yes.
        => Use dynamic programming
        => Complexity falls to O(N**2) and beats 40%
        """

        def cached(f):
            memo = {}

            def wrapped(*args):
                if args in memo:
                    return memo[args]
                res = f(*args)
                memo[args] = res
                return res

            return wrapped

        @cached
        def visit(start: int) -> int:
            if start == n:
                return 0

            min_ops = float('inf')
            copy_times = 1
            while start + copy_times * start <= n:
                min_ops = min(min_ops, copy_times + visit(start + copy_times * start))
                copy_times += 1
            return 1 + min_ops  # The 1 is for the original copy

        return visit(1)
