"""
https://practice.geeksforgeeks.org/problems/largest-fibonacci-subsequence/0

Given an array with positive number the task to find the largest sub-sequence from array that contain elements
which are Fibonacci numbers.
"""


from typing import List


def fibonacci_up_to(n):
    a = 0
    b = 1
    fibs = {0}
    while b <= n:
        fibs.add(b)
        a, b = b, a + b
    return fibs


def largest_fib_subsequence(nums: List[int]) -> List[int]:
    if not nums:
        return []

    fibs = fibonacci_up_to(max(nums))
    return [x for x in nums if x in fibs]


print(largest_fib_subsequence([0, 2, 8, 5, 2, 1, 4, 13, 23]))
