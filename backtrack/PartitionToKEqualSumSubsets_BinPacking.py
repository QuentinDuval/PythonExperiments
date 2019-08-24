"""
https://leetcode.com/problems/partition-to-k-equal-sum-subsets/

Given an array of integers nums and a positive integer k, find whether it's possible to divide this array into
k non-empty subsets whose sums are all equal.
"""


from typing import List


class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        """
        This is Bin Packing, this is NP Hard.
        => No way but to try everything: backtracking.

        Beats 97% (40 ms)
        """

        target, rest = divmod(sum(nums), k)
        if rest != 0:
            return False

        # Sort the numbers to eliminate as many case as early as possible
        nums.sort(reverse=True)

        def visit(pos: int, partitions: List) -> int:
            if pos == len(nums):
                return True

            possible = False
            for i in range(len(partitions)):
                if partitions[i] + nums[pos] <= target:
                    partitions[i] += nums[pos]
                    possible = visit(pos + 1, partitions)
                    partitions[i] -= nums[pos]
                    if possible:
                        return True
                    # If the last partition tried was 0, remaining are also 0, so cut here
                    if partitions[i] == 0:
                        break
            return possible

        return visit(0, [0] * k)
