"""
https://practice.geeksforgeeks.org/problems/decode-the-pattern/0
"""

from typing import *


def next_look_and_say(seq: List[int]) -> List[int]:
    next_seq = []
    count = 1
    prev = seq[0]
    for num in seq[1:]:
        if num == prev:
            count += 1
        else:
            next_seq.append(count)
            next_seq.append(prev)
            prev = num
            count = 1
    next_seq.append(count)
    next_seq.append(prev)
    return next_seq


def look_and_say(n: int):
    seq = [1]
    for _ in range(1, n):
        seq = next_look_and_say(seq)
    return seq
