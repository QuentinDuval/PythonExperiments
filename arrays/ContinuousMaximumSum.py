"""
https://practice.geeksforgeeks.org/problems/kadanes-algorithm/0

Given an array arr of N integers. Find the contiguous sub-array with maximum sum.
"""


from typing import List


def continuous_maximum_sum(nums: List[int]) -> int:
    """
    The continuous range could contain negative number (fine)
    The continuous range might contain negative continuous range (fine)

    Start at index 'i' and sum contiguous elements
    If a range started at 'i' goes negative at index 'j':
    - It is useless to start again at index 'k' s.t. 'i' < 'k' < 'j'
      (the continuous sum was positive, you can only remove to it - see integral)
    - So you have to start again at the index 'j' + 1
    - Of course keep the maximum you found anyway

    Example: [1, 2, 3, -3, -4, 5, -1, 3] => 7
    [1, 2, 3, -3, -4, 5, -1, 3]
     ^             ^
    [1, 2, 3, -3, -4, 5, -1, 3]
                      ^      ^

    Note: you could found this by just trying a more naive algorithm and see that it is bound to fail trying k s.t. i < k < j.
    """
    if not nums:
        return 0

    best_sum = nums[0]
    cum_sum = 0
    for n in nums:
        cum_sum += n
        if cum_sum < 0:
            cum_sum = 0
        else:
            best_sum = max(best_sum, cum_sum)
    return best_sum
