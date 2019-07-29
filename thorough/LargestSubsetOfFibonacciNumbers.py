"""
https://practice.geeksforgeeks.org/problems/largest-fibonacci-subsequence/0

Given an array with positive number the task to find the largest sub-sequence from array
that contain elements which are Fibonacci numbers.
"""

from typing import List


def fibonacci_until(n: int):
    fibs = [0, 1]
    while fibs[-1] < n:
        fibs.append(fibs[-2] + fibs[-1])
    return set(fibs)


def fibonacci_sub_seq(nums: List[int]) -> List[int]:
    max_val = max(nums, default=1)
    fibs = fibonacci_until(max_val)
    sub_seq = []
    for num in nums:
        if num in fibs:
            sub_seq.append(num)
    return sub_seq


t = int(input())
for _ in range(t):
    n = int(input())
    nums = [int(n) for n in input().split()[:n]]
    res = fibonacci_sub_seq(nums)
    print(" ".join(str(n) for n in res))


