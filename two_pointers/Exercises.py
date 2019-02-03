from typing import List


"""
Problem from leetcode:
https://leetcode.com/problems/container-with-most-water/
"""


def water_max_area(heights: List[int]) -> int:
    """
    The way to do it in brute force would be to try every pair
    of points i < j and check the max area.

    But this is clearly wasteful:
    - From the left: No need to consider point j if h(i) >= h(j) with i < j
    - From the right: No need to consider point i if h(i) <= h(j) with i < j

    This suggests a solution that starts from both ends.
    """

    """
    BAD SOLUTION:

    Keep a sorted list, constructed from left to right, of the
    max height available at the left of each point.

    Keep a sorted list, constructed from right to left, of the
    max height available at the right of each point.
    
    For each point look to left and right...
    
    NO! It would ignore something like this:
    [1,1,1,1,1,1,2,2,1,1,1,1,1]
    """

    '''
    n = len(heights)

    def left_levels(order):
        levels = []
        for i in order:
            h = heights[i]
            prev = levels[-1][1] if levels else 0
            if h > prev:
                levels.append((i, h))
            else:
                levels.append(levels[-1])
        return levels

    lefts = left_levels(range(n))
    rights = left_levels(reversed(range(n)))[::-1]
    '''

    """
    GOOD SOLUTION WITH JUST TWO POINTERS:
    - Move left if left is lower than right
    - Move right if right is lower than right
    """

    max_area = 0

    lo = 0
    hi = len(heights) - 1
    while lo <= hi:
        if heights[lo] < heights[hi]:
            max_area = max(max_area, heights[lo] * (hi - lo))
            lo += 1
        else:
            max_area = max(max_area, heights[hi] * (hi - lo))
            hi -= 1
    return max_area

input = [1,8,6,2,5,4,8,3,7]
print(water_max_area(input))


"""
3 sum closest:
https://leetcode.com/problems/3sum-closest
Given an array of number, find a triplet whose sum is closest to 'target'
"""


def three_sum_closest(nums: List[int], target: int) -> int:
    """
    The key is to rely on sorting
    - Move a first pointer 'neg' through the list in increasing order
    - For each 'neg' value:
      - Initialize a "low" pointer at 'neg + 1'
      - Initialize a "high" pointer at 'len(nums) - 1'
      - If the sum at these three positions is higher than target, move 'high' to left
      - Else move 'low' to right

    Why does this work?
    Just consider that if 'low' gets bigger, 'high' has to go smaller to become closer to the target
    This is what you would expect from a binary search point of view.
    """
    n = len(nums)
    nums.sort()

    closest = sum(nums[:3])
    for neg in range(n - 2):
        low, high = neg + 1, n - 1
        while low < high:
            triplet_sum = nums[neg] + nums[low] + nums[high]
            closest = triplet_sum if abs(triplet_sum - target) < abs(closest - target) else closest
            if triplet_sum == target:
                return target
            elif triplet_sum > target:
                high -= 1
                while low < high and nums[high] == nums[high + 1]:
                    high -= 1
            else:
                low += 1
                while low < high and nums[low - 1] == nums[low]:
                    low += 1

    return closest

