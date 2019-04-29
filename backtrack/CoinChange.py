from functools import *

"""
Minimum coin change (an optimization problem):
https://leetcode.com/problems/coin-change/
The backtracking one beats the hell out of the memoization implementations
"""


def min_coin_change_dp_top_bottom(coins, amount):
    @lru_cache(maxsize=None)
    def visit(remaining):
        if remaining == 0: return 0
        if remaining < 0: return float('inf')
        return 1 + min((visit(remaining-coin) for coin in coins if coin <= amount), default=float('inf'))
    return visit(amount)


def min_coin_change_dp_bottom_up(coins, amount):
    memo = [float('inf')] * (amount + 1)
    memo[0] = 0
    for remaining in range(1, amount+1):
        subs = (memo[remaining-coin] for coin in coins if coin <= amount)
        memo[remaining] = 1 + min(subs, default=float('inf'))
    return memo[-1]


def min_coin_change(coins, amount):
    if amount == 0:
        return 0

    coins.sort(reverse=True)

    def backtrack(remaining, i, current_best):
        if i >= len(coins):
            return float('inf')

        coin = coins[i]
        if coin > remaining:
            return backtrack(remaining, i + 1, current_best)

        q, r = divmod(remaining, coin)
        if r == 0:
            return q

        if q >= current_best:
            return float('inf')

        min_res = current_best
        while q >= 0:
            sub_res = q + backtrack(r, i + 1, current_best=min_res - q)
            min_res = min(min_res, sub_res)
            r += coin
            q -= 1

        return min_res

    return backtrack(remaining=amount, i=0, current_best=float('inf'))


"""
Combinations of coin changes (a pure combinatorial problem + memoization):
https://leetcode.com/problems/coin-change-2/
"""


def count_change(coins, amount):

    """
    # This does not work because we have many equivalent combinations
    for remaining in range(1, amount+1):
        count = 0
        for coin in coins:
            if coin <= remaining:
                count += memo[remaining - coin]
        memo[remaining] = count

    return memo[-1]
    """

    """
    # This correctly works:
    # - we account for each coin just once
    # - we try all count for that coin
    # But it is slow and can be simplified
    memo = [0] * (amount + 1)
    memo[0] = 1

    for coin in coins:
        new_memo = [0] * (amount + 1)
        new_memo[0] = 1

        for remaining in range(1, amount+1):
            count = 0
            max_q = remaining // coin
            for q in range(max_q+1):
                count += memo[remaining-q*coin]
            new_memo[remaining] = count
        memo = new_memo

    return memo[-1]
    """

    # This version is vastly simplified:
    # By taking the remaining in the right order, we profit from memoization furthermore
    # And we having having to consider every single quotient from 0 to remaining // coin

    memo = [0] * (amount + 1)
    memo[0] = 1
    for coin in coins:
        for remaining in range(coin, amount + 1):
            memo[remaining] += memo[remaining - coin]
    return memo[-1]
