from collections import Counter
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

