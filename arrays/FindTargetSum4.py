"""
Form with TWO NUMBERS:

Given an array A of N elements.
Complete the function which returns true if 4-sum exists in array A whose sum is target else return false.
"""


from typing import List


def find_target_sum_4(nums: List[int], target: int):
    """
    Solution attempt:
    Compute the sum of all pairs, then do a find_target_sum_2.
    => Complexity is O(N**2) if we use the find_target_sum_2 with hashing.

    BUT IT DOES NOT WORK (duplicates could be used...)
    """
    pair_sums = [nums[i] + nums[j] for i in range(len(nums) - 1) for j in range(i, len(nums))]
    # TODO
