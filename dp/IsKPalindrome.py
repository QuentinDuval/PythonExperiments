"""
https://practice.geeksforgeeks.org/problems/k-palindrome/1

A string is k palindrome if it can be transformed into a palindrome on removing at most k characters from it.
Your task is to complete the function is_k_palin which takes two arguments a string str and a number N .
Your function should return true if the string is k palindrome else it should return false.
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


def is_k_palin(s: str, n: int):
    @cached
    def is_k_palin_range(lo: int, hi: int, n: int):
        if n < 0:
            return False
        if lo >= hi:
            return True
        if s[lo] == s[hi]:
            return is_k_palin_range(lo + 1, hi - 1, n)
        return is_k_palin_range(lo + 1, hi, n - 1) or is_k_palin_range(lo, hi - 1, n - 1)

    return is_k_palin_range(0, len(s) - 1, n)
