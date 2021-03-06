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
Generate parenthesis: https://leetcode.com/problems/generate-parentheses/
Generate all combination of valid parenthesis for size N
"""


def generateParenthesis(n: int) -> List[int]:
    """
    The first challenge here is to generate each combination once and just once

    The second challenge is to reuse sub-problems (memoization) also it is less
    useful (because the solutions will be copied for each combinations anyway)

    One way to generate unique pair is to rely on the recurrence of catalan numbers.
    To solve the problem for N, named G(N):
    - Solve the problem for G(K) and wrap it in parentheses
    - Solve the problem for G(N-1-K) (do not wrap it)
    - Combine these sub-problems for each K in 1..N-1

    This can be seen as:
    1) Choosing when to close the parenthesis we first opened
    2) Recurring on each side of this parenthesis

    BEWARE: if you just recurse for K and N-K and not wrap parenthesis, you just
    double count solutions such as ()()().
    """

    '''
    def backtrack(n):
        if n == 0: return [""]

        solutions = []
        for k in range(0,n):
            for left in backtrack(k):
                for right in backtrack(n-1-k):
                    solutions.append("(" + left + ")" + right)
        return solutions
    return backtrack(n)
    '''

    """
    The second way to generate all combinations just once is to do
    a backtracking based on keeping a count of open and closed parenthesis
    """
    partial = []
    solutions = []

    def backtrack(opened, closed):
        if opened == n:
            solutions.append("".join(partial) + ")" * (opened - closed))
            return

        partial.append("(")
        backtrack(opened + 1, closed)
        partial.pop()

        if closed < opened:
            partial.append(")")
            backtrack(opened, closed + 1)
            partial.pop()

    backtrack(0, 0)
    return solutions


"""
Generate all parenthesisation of an arithmetic expression
"""


# TODO


"""
Exercise 7.10 of "The algorithm design manual"
Solve the minimum vertex coloring problem:
- Return the minimum number of colors needed to color each vertex
- Such that no two adjacent vertices have the same color
"""


# TODO


"""
Who wins?
"""


# TODO - game https://leetcode.com/problems/nim-game/
# TODO - game https://leetcode.com/problems/can-i-win/
# TODO - game https://leetcode.com/problems/predict-the-winner/


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
COMBINATION SUM: https://leetcode.com/problems/combination-sum/
- Return all combinations of the list of candidates that sum to the target
- Each number can be used as many times as desired
"""


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    This is quite similar to the coin change problem:
    How to exchange a given amount of money with the provided coins

    The key aspect is to avoid double counting:
    Consider the candidates in order to avoid having [2,2,3] and [2,3,2] for instance

    Optimizations:
    - Sorting the candidates allows to prune early, so it is a good bargain
    - Memoization is not that useful (sub-solutions will be duplicated in terms of memory anyway)
    => This is what makes it a backtracking problem more than a dynamic programming problem
    """
    candidates.sort()

    partial = []
    solutions = []

    def backtrack(remaining, position):
        if remaining == 0:
            solutions.append(list(partial))

        if position >= len(candidates):
            return

        q, r = divmod(remaining, candidates[position])
        if q == 0:
            return

        partial.extend([candidates[position]] * q)

        while True:
            backtrack(r, position + 1)
            r += candidates[position]
            q -= 1
            if q >= 0:
                partial.pop()
            else:
                break

    backtrack(remaining=target, position=0)
    return solutions

    """
    Equivalent implementation (perhaps less obvious for the non duplication of candidates)
    """

    '''
    candidates.sort()
    partial = []

    def visit(start_candidate, target):
        if target == 0:
            yield list(partial)
            return

        for i in range(start_candidate, len(candidates)):
            c = candidates[i]
            if c > target:
                break

            partial.append(c)
            yield from visit(start_candidate=i, target - c)
            partial.pop()

    return list(visit(0, target))
    '''

    """
    Try with memoization
    """

    # TODO


"""
COMBINATION SUM 2: https://leetcode.com/problems/combination-sum-2/
- Return all combinations of the list of candidates that sum to the target
- Each number can be used at much ONE TIME
"""


def combination_sums_2(candidates: List[int], target: int) -> List[List[int]]:
    """
    Backtracking:
    - Avoid visiting the same solutions twice by grouping number
    - Alternatively, you can skip equal number when ignore the first one

    Possible optimizations:
    - Sorting the number allows to prune quickly (works)
    - Keeping a cumulative right sum to prune quickly (almost no effect)
    """

    cumulative_sum = sum(candidates)
    if cumulative_sum < target: return []
    if cumulative_sum == target: return [candidates]
    candidates = list(sorted(Counter(candidates).items()))

    partial = []
    solutions = []

    def backtrack(remaining, position):
        if remaining == 0:
            solutions.append(list(partial))

        if position == len(candidates):
            return

        candidate, count = candidates[position]
        if candidate > remaining:
            return

        count = min(count, remaining // candidate)
        partial.extend([candidate] * count)
        while count >= 0:
            backtrack(remaining - candidate * count, position + 1)
            if count > 0:
                partial.pop()
            count -= 1

    backtrack(target, 0)
    return solutions

