"""
https://practice.geeksforgeeks.org/problems/k-palindrome/1

A string is k palindrome if it can be transformed into a palindrome on removing at most k characters from it.
Your task is to complete the function is_k_palin which takes two arguments a string str and a number N .
Your function should return true if the string is k palindrome else it should return false.

Example:

    abcdecba 1  => true
    acdcb  1    => false

String is of size 100 at maximum, K of size 20 at maximum.
"""


"""
How to proceed?
---------------

We could start on both end, and try to remove annoying letters (on conflict):
- 'abcdecba' => remove 'd' or 'e' (try both possibles)

We could remove letters on one end:
- 'abcdcbae' => 'e' (you have to try both possibles)

In both case, it is at the first conflict that you must check both possibilities.
Backtracking => Check overlapping problems (there are some, check 'eabcdcbaf')
"""


def is_k_palin(s: str, k: int) -> bool:
    """
    First approach, based on recursion and memoization
    """
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def visit(lo, hi, n):
        while lo <= hi and s[lo] == s[hi]:
            lo += 1
            hi -= 1

        if lo >= hi:
            return True
        elif n == 0:
            return False
        else:
            return visit(lo, hi - 1, n - 1) or visit(lo + 1, hi, n - 1)

    return visit(0, len(s) - 1, k)


print(is_k_palin('abcdecba', 1))
print(is_k_palin('acdcb', 1))
