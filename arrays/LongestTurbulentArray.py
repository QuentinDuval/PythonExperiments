"""
https://leetcode.com/problems/longest-turbulent-subarray

A subarray A[i], A[i+1], ..., A[j] of A is said to be turbulent if and only if the comparison sign flips between
each adjacent pair of elements in the subarray.

Return the length of a maximum size turbulent subarray of A.
"""


from typing import List


class Solution:
    def maxTurbulenceSize(self, nums: List[int]) -> int:
        """
        Focus on the case where odd indices are superior to the even indices.

        We can easily realize that it is enough to do a simple scan:
        - if a sub-array [i:j] is turbulent and [j:k] is turbulent, their concatenation is turbulent
        (this does not work if they have 'different turbulence')

        => we can do 2 scans for a O(N) algorithm
        """

        if len(nums) <= 1:
            return len(nums)

        def scan_odd(odd_incr):
            max_len = 1
            curr_len = 1
            for i in range(0, len(nums) - 1):
                if i & 1 == odd_incr:
                    if nums[i] < nums[i + 1]:
                        curr_len += 1
                        max_len = max(max_len, curr_len)
                    else:
                        curr_len = 1
                else:
                    if nums[i] > nums[i + 1]:
                        curr_len += 1
                        max_len = max(max_len, curr_len)
                    else:
                        curr_len = 1
            return max_len

        return max(scan_odd(0), scan_odd(1))
