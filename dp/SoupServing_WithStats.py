"""
https://leetcode.com/problems/soup-servings

There are two types of soup: type A and type B. Initially we have N ml of each type of soup. There are four kinds of operations:

Serve 100 ml of soup A and 0 ml of soup B
Serve 75 ml of soup A and 25 ml of soup B
Serve 50 ml of soup A and 50 ml of soup B
Serve 25 ml of soup A and 75 ml of soup B
When we serve some soup, we give it to someone and we no longer have it.  Each turn, we will choose from the four operations with equal probability 0.25. If the remaining volume of soup is not enough to complete the operation, we will serve as much as we can.  We stop once we no longer have some quantity of both types of soup.

Note that we do not have the operation where all 100 ml's of soup B are used first.

Return the probability that soup A will be empty first, plus half the probability that A and B become empty at the same time.
"""

import numpy as np


class Solution:
    def soupServings(self, N: int) -> float:

        # Basic statistics on the different kind of recipees
        A = np.array([100, 75, 50, 25])
        B = np.array([0, 25, 50, 75])
        mean_a = np.mean(A)
        mean_b = np.mean(B)
        sigma = np.sqrt(np.mean(A ** 2) - mean_a ** 2)  # Same for A and B

        # At 3 sigma, you have 0.1% of chance => 1e-6 for both A and B
        avg_turns = N / 50
        lo_avg_a = avg_turns * mean_a - 4 * np.sqrt(avg_turns) * sigma
        hi_avg_b = avg_turns * mean_b + 4 * np.sqrt(avg_turns) * sigma
        if lo_avg_a > hi_avg_b:
            return 1.0

        # Caching mecanism for memoization
        def cache(f):
            memo = {}

            def new_f(*args):
                res = memo.get(args)
                if res is not None:
                    return res
                res = f(*args)
                memo[args] = res
                return res

            return new_f

        @cache
        def visit(a: int, b: int) -> float:
            if a <= 0 and b > 0:
                return 1
            elif a <= 0 and b <= 0:
                return 0.5
            elif a > 0 and b <= 0:
                return 0
            p = 0
            for i in range(4):
                p += 0.25 * visit(a - A[i], b - B[i])
            return p

        return visit(N, N)
