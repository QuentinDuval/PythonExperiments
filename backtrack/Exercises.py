from collections import Counter
import bisect
import functools
import itertools
from typing import *


"""
Exercise 7.1 of "The algorithm design manual"
Write a function to (via backtrack / pruning) find all derangement of a set [1..n]
A derangement is such that pi != i for all i in [1..n] 
"""


def derangement(n):
    partial = []
    remaining = set(range(1, n+1))
    solutions = []

    def backtrack(i):
        if i > n:
            solutions.append(list(partial))
            return

        for p in list(remaining):
            if p != i:
                remaining.remove(p)
                partial.append(p)
                backtrack(i+1)
                partial.pop()
                remaining.add(p)

    backtrack(1)
    return solutions


"""
Exercise 7.1 of "The algorithm design manual"
Write a function to find all the unique permutations of a multi-set (several occurrences of the same number)
"""


def multiset_permutations(multiset):
    n = len(multiset)
    partial = []
    remaining = Counter(multiset)   # The key is to group elements to avoid duplicates
    solutions = []

    def backtrack(i):
        if i > n:
            solutions.append(list(partial))
            return

        for v, count in remaining.items():
            if count > 0:
                remaining[v] -= 1
                partial.append(v)
                backtrack(i+1)
                partial.pop()
                remaining[v] += 1

    backtrack(1)
    return solutions


"""
Exercise 7.10 of "The algorithm design manual"
Solve the minimum vertex coloring problem:
- Return the minimum number of colors needed to color each vertex
- Such that no two adjacent vertices have the same color
"""


# TODO


"""
Exercise 7.17 of "The algorithm design manual"
Generate all possible words from translating a given digit sequence of a telephone keypad
"""

keypad_letters = {
    1: "",
    2: "abc",
    3: "def",
    4: "ghi",
    5: "jkl",
    6: "mno",
    7: "pqrs",
    8: "tuv",
    9: "wxyz",
    0: " "
}


def keypad_words(digits, dictionary) -> List[str]:
    n = len(digits)
    partial = []
    solutions = []

    def backtrack(i):
        if i >= n:
            if dictionary.is_member(partial):
                solutions.append(list(partial))
            return

        for letter in keypad_letters[digits[i]]:
            partial.append(letter)
            if dictionary.is_prefix(partial):
                backtrack(i+1)
            partial.pop()

    backtrack(0)
    return ["".join(solution) for solution in solutions]


"""
Minimum coin change:
https://leetcode.com/problems/coin-change/
The backtracking one beats the hell out of the memoization implementations
"""


def min_coin_change_dp_top_bottom(coins, amount):
    @functools.lru_cache(maxsize=None)
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
Combinations of coin changes:
https://leetcode.com/problems/coin-change-2/
"""

