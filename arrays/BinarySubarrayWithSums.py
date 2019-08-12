"""
https://leetcode.com/problems/binary-subarrays-with-sum/

In an array A of 0s and 1s, how many non-empty subarrays have sum S?
"""


from typing import List


class Solution:
    def numSubarraysWithSum(self, nums: List[int], target: int) -> int:
        """
        We can at the very least implement it in O(N ^ 2):
        - try for all 0 <= i < j < N
        - precompute the prefix_sum in O(N)

        We can probably try a window based approach:
        - Grow the window when the sum is lower (and there are more zeros to the right)
        - Diminish the window when the sum is higher (and add the number of zeros to the right + 1 to the count)
        """
        if target == 0:
            return self.numSubArraySumZero(nums)

        count = 0
        lo = 0  # Points at the beginning of the window
        hi = 0  # Points after the end of the window
        window = 0

        while lo < len(nums):
            # Extend the window until we reach the window
            while hi < len(nums) and window < target:
                window += nums[hi]
                hi += 1

            if window < target:
                break

            # Extend the window with all zeros to the right
            right_zeros = 0
            while hi < len(nums) and nums[hi] == 0:
                right_zeros += 1
                hi += 1

            # Count the number of solutions
            while lo < hi and nums[lo] == 0:
                count += (1 + right_zeros)
                lo += 1
            count += (1 + right_zeros)
            lo += 1
            window -= 1

        return count

    def numSubArraySumZero(self, nums: List[int]) -> int:
        count = 0
        lo = 0  # Points at the beginning of the window
        while lo < len(nums):
            while lo < len(nums) and nums[lo] != 0:
                lo += 1
            hi = lo
            while hi < len(nums) and nums[hi] == 0:
                hi += 1
            l = (hi - lo)
            count += (l + 1) * l // 2
            lo = hi
        return count

