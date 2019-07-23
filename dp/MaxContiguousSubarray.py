"""
Find the contiguous sub-array whose sum is maximized
"""
from typing import List


def max_sum_sub_array(nums: List[int]):
    """
    Naive algorithm is O(N ** 3): try all i < j and sum in between.
    Improvement with computation of prefix sum yields O(N ** 2).

    Better solution is to reason that we always want to continue a sum that is positive.
    - So we start a new contiguous sum each time the previous prefix sum goes below 0
    - And we keep track of the maximum sum we found so far
    """

    max_sum = -float('inf')
    curr = 0
    for num in nums:
        if curr < 0:
            curr = num
        else:
            curr += num
        max_sum = max(max_sum, curr)
    return max_sum


print(max_sub_array([-4, 3, -2, 3, -3]))
