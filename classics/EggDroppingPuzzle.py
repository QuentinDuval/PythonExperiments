"""
https://practice.geeksforgeeks.org/problems/egg-dropping-puzzle/0

Suppose you have N eggs and you want to determine from which floor in a K-floor building you can drop an egg such that it doesn't break.
You have to determine the minimum number of attempts you need in order find the critical floor in the worst case.
"""


"""
If you drop an egg at floor h:
- Either it breaks: you have to solve the problem with one less egg and h - 1 floors
- Or it survives: you have to solve the problem with same amount of eggs and floor - h floors
=> You want to select the height h such that both costs are equal (the worst cost is the worst of both)

Solution using DP:
- Number of sub-problems is O(eggs * floors)
- Space complexity is O(eggs * floors)
- Time complexity is O(eggs * floors * floors)
"""


def min_attempts(eggs, floors):
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def visit(eggs, floors):
        if eggs == 1:
            return floors

        if floors == 0:
            return 0

        return min(
            1 + max(visit(eggs-1, h-1), visit(eggs, floors-h))
            for h in range(1, floors+1)
        )

    return visit(eggs, floors)


"""
This time using bottom-up DP
- Space complexity is down to O(floors)
"""


def min_attempts_2(eggs, floors):
    if eggs <= 1:
        return floors

    memo = list(range(floors + 1))

    for _ in range(1, eggs):
        new_memo = [0] * (floors + 1)
        new_memo[1] = 1
        for sub_floor in range(2, floors+1):
            new_memo[sub_floor] = min(
                1 + max(memo[h - 1], new_memo[sub_floor - h])
                for h in range(1, sub_floor+1)
            )
        memo = new_memo

    return memo[-1]


# print(min_attempts_2(2, 10))
# print(min_attempts_2(2, 100))
