from typing import List


"""
https://leetcode.com/problems/minimum-moves-to-equal-array-elements-ii

Given a non-empty integer array, find the minimum number of moves required to make all array elements equal,
where a move is incrementing a selected element by 1 or decrementing a selected element by 1.
"""


def minMoves2(nums: List[int]) -> int:
    """
    Why would a simple difference to the average not work?
    Here is a counter example:
    [1,4,4] => mean is 3 => 1 + 1 + 2 while best solution is 3

    Why would a simple difference to the median not work?
    [1,3,4,4] => median is 3 or 4... both cases lead to 4, the correct answer
    [1,2,4,4] => median is 2 or 4... both cases load to 5, the correct answer

    It actually works, with both lower and upper median!
    """
    if len(nums) < 1:
        return 0

    nums.sort()
    mid_val = nums[len(nums) // 2]
    return sum(abs(mid_val - n) for n in nums)
