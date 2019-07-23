"""
https://practice.geeksforgeeks.org/problems/largest-sum-subarray-of-size-at-least-k/0

Given an array and a number k, find the largest sum of the subarray containing at least k numbers.
It may be assumed that the size of array is at-least k.
"""
from typing import List


def max_sum_sub_array_with_k_numbers(nums: List[int], k: int):
    """
    We know how to compute the max sum of a contiguous sub-array
    - we cannot use a sliding window approach directly on the 'nums' array: how do we know when to extend?
    - we could resort to an O(N ** 2) algorithm (all pairs i < j plus prefix sum)

    But we can do better:
    - try each end of range of contiguous sub-array, and find the best sum that ends at that position
      (same algo as max sum sub-array until each and every position)
    - use this knowledge to find the best sub-array sum of K elements ending at each and every position
      (compute the sum of each range of K elements using prefix sums, then add to best sum at beginning of range)
    (https://www.geeksforgeeks.org/largest-sum-subarray-least-k-numbers)
    """

    prefix_sum_at = [0]
    max_sum_ending_at = [0]
    for i, num in enumerate(nums):
        prefix_sum_at.append(prefix_sum_at[-1] + num)
        max_sum_ending_at.append(max(num, max_sum_ending_at[-1] + num))

    print(prefix_sum_at)
    print(max_sum_ending_at)

    max_sum = float('-inf')
    for i in range(k, len(max_sum_ending_at)):
        k_sum = prefix_sum_at[i] - prefix_sum_at[i - k]
        before = max_sum_ending_at[i - k]
        max_sum = max(max_sum, k_sum, k_sum + before)
    return max_sum


nums = [int(n) for n in "5 7 -9 3 -4 2 1 -8 9 10".split(" ")]
k = 5
print(max_sum_sub_array_with_k_numbers(nums, k))

