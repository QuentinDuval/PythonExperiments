from typing import *


def findSubstringInWraproundString(p: str) -> int:
    """
    https://leetcode.com/problems/unique-substrings-in-wraparound-string

    Find all substrings in p that are also substrings of the wrap around string:
    "...zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd...."

    These substrings are necessarily following the increasing order of the alphabet
    """

    n = len(p)
    if n == 0:
        return 0

    def next_char(c: str) -> str:
        if c == 'z':
            return 'a'
        return chr(ord(c) + 1)

    found = {}

    def add_to_found(lo, hi):
        for i in range(lo, hi):
            c = p[i]
            found[c] = max(found.get(c, 0), hi - i)

    start_interval = 0
    prev = p[0]
    for end_interval in range(1, n):
        if next_char(prev) != p[end_interval]:
            add_to_found(start_interval, end_interval)
            start_interval = end_interval
        prev = p[end_interval]

    add_to_found(start_interval, n)
    return sum(found.values())
