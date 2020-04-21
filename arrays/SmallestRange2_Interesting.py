"""
https://leetcode.com/problems/smallest-range-ii/

Given an array A of integers, for each integer A[i] we need to choose either x = -K or x = K,
and add x to A[i] (only once).

After this process, we have some array B.

Return the smallest possible difference between the maximum value of B and the minimum value of B.
"""


from typing import List


class Solution:
    def smallestRangeII(self, nums: List[int], k: int) -> int:
        """
        Observations:
        - the order of A does not matter => sorting it will likely help
        - increasing all by K will not change the value

        We increase all values of nums by K (in fact we do nothing as relative stuff)
        - we then sort the number in decreasing order
        - we search for the best point i such that nums[:i] gets substracted 2K, and nums[i:] is kept

        To do so we start with the highest values and substract 2K
        - we keep track of the 'hi' and 'lo' of the range
        - we stop when there is no improvement possible ('hi' cannot decrease)
        """

        hi = max(nums)
        lo = min(nums)
        nums.sort(reverse=True)

        min_gap = hi - lo
        for i in range(len(nums) - 1):
            next_val = nums[i] - 2 * k
            next_hi = max(nums[i + 1], nums[0] - 2 * k)
            next_lo = min(lo, nums[i] - 2 * k)

            if next_val >= nums[i + 1]:
                min_gap = min(min_gap, next_hi - next_lo)
                break

            hi, lo = max(next_hi, next_lo), min(next_hi, next_lo)
            min_gap = min(min_gap, hi - lo)
        return min_gap
