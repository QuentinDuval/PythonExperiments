"""
https://practice.geeksforgeeks.org/problems/egg-dropping-puzzle/0

Suppose you have N eggs and you want to determine from which floor in a K-floor building you can drop an egg such that it doesn't break.
You have to determine the minimum number of attempts you need in order find the critical floor in the worst case.
"""


import math
from functools import lru_cache


class Solution:
    def superEggDrop(self, eggs: int, floors: int) -> int:
        """
        If you drop an egg at floor h:
        - Either it breaks: you have to solve the problem with one less egg and h - 1 floors
        - Or it survives: you have to solve the problem with same amount of eggs and floor - h floors
        => You want to select the height h such that both costs are equal (the worst cost is the worst of both)

        Recursive formula:

        superEggDrop(eggs=1, floors=N) -> N

        superEggDrop(eggs=2, floors=N) ->

            We have an exact formula here: if we throw at floor F, we must equilibrate the two cases:
            - the egg does break and we have to do F-1 tries after
            - the egg does not break and we have F-2 tries after (if it breaks right after) or F-3, etc.

            => Find F such that:
            F(F-1) / 2 >= N
            F ** 2 - F - 2*N = 0
            (F - 1/2) ** 2 - 1/4 - 2*N = 0
            (F - 1/2) ** 2 = 1/4 (8N + 1)
            F = 1/2 +- 1/2 * sqrt(8N + 1)
            F = 1/2 +- 1/2 * sqrt(8N + 1)

            Other way to find it was:
            ax^2 + bx + c = 0 => x = 1 / 2a * (-b +- sqrt(b**2 - 4ac))

        superEggDrop(eggs=K, floors=N) ->
            min(
                1 +
                max(superEggDrop(K-1, floors=X-1), # does break and you have to look under
                    superEggDrop(K, floors=N-X)) # does not break and you have to look above
                for X in range(1, N)
            )

        Use memoization to avoid recomputing the same problems over and over again.
        - Number of sub-problems: N * K
        - Work by sub-problems: O(N)

        Time complexity: O(N * N * K) => TIMEOUT at eggs=4, floors=5000
        Space complexity: O(N * K) with top-down
        """

        @lru_cache(maxsize=None)
        def minimize(eggs: int, floors: int) -> int:
            if eggs == 1 or floors <= 1:
                return floors

            if eggs == 2:
                return math.ceil(0.5 + 0.5 * math.sqrt(8 * floors + 1)) - 1

            return min(
                1 + max(minimize(eggs - 1, floor - 1), minimize(eggs, floors - floor))
                for floor in range(1, floors))

        return minimize(eggs, floors)


"""
Bottom-up memoization:
Time complexity: O(N * N * K)
Space complexity: O(N)

 => TIMEOUT at eggs=4, floors=5000
"""


class Solution:
    def superEggDrop(self, eggs: int, floors: int) -> int:
        if eggs == 1 or floors <= 1:
            return floors

        if eggs == 2:
            return math.ceil(0.5 + 0.5 * math.sqrt(8 * floors + 1)) - 1

        memo = list(range(floors + 1))
        memo[1] = 1
        for floor in range(2, floors + 1):
            memo[floor] = math.ceil(0.5 + 0.5 * math.sqrt(8 * floor + 1)) - 1

        for _ in range(eggs - 2):
            new_memo = [0] * (floors + 1)
            new_memo[1] = 1
            for sub_floor in range(2, floors + 1):
                new_memo[sub_floor] = min(
                    1 + max(memo[h - 1], new_memo[sub_floor - h])
                    for h in range(1, sub_floor + 1)
                )
            memo = new_memo

        return memo[-1]

# TODO - save from where you came from...


# print(min_attempts_2(2, 10))
# print(min_attempts_2(2, 100))
