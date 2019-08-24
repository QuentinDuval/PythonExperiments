"""
https://leetcode.com/problems/split-array-into-consecutive-subsequences/

Given an array nums sorted in ascending order, return true if and only if you can split it into 1 or
more subsequences such that each subsequence consists of consecutive integers and has length at least 3.
"""

from typing import List


class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        """
        [1, 2, 3, 4, 5, 6]
        => [1, 2, 3] + [4, 5, 6] or [1, 2, 3, 4, 5, 6]
        (no need to split if not needed)

        [1, 2, 3, 3, 3, 4, 4, 5, 5, 6]
        => [1, 2, 3] + [3, 4, 5] + [3, 4, 5, 6]

        [1, 2, 3x3, 4x2, 5x2, 6]
        => Based on this, we can see the sequences

        The principle is the following (Greedy):
        - go from left to right and try to expand existing strands
        - prioritize the shortest strands (with a heap)
        - if there is no strand to extend, create a new strand
        - skip the strand and freeze it (get rid of it) if there is a gap
        - return False if the froozen strand is of size below 3
        """

        def run_length_encoding(nums):
            count = 1
            prev = nums[0]
            for num in nums[1:]:
                if num == prev:
                    count += 1
                else:
                    yield prev, count
                    count = 1
                    prev = num
            yield prev, count

        prev_num = None  # The previous number encountered
        strands = [0, 0, 0]  # The number of strands of length 1, 2 and more than 3

        for num, count in run_length_encoding(nums):

            # We cannot continue the previous strands (not enough number to extend them)
            if strands[0] + strands[1] > count:
                return False

            next_strands = [0, 0, 0]
            if prev_num and prev_num != num - 1:
                # There is a gap and we cannot continue the previous strands
                if strands[0] != 0 or strands[1] != 0:
                    return False
                # Start new strands otherwise
                next_strands[0] = count
            else:
                # If there is not gap, extend the strands of size 1, then 2, then 3
                # If there are remaining numbers, create new strands of size 1
                next_strands[1] = strands[0]
                count -= strands[0]
                next_strands[2] = strands[1]
                count -= strands[1]
                if count > strands[2]:
                    next_strands[2] += strands[2]
                    next_strands[0] = count - strands[2]
                else:
                    next_strands[2] += count

            strands = next_strands
            prev_num = num

        return strands[0] == 0 and strands[1] == 0


