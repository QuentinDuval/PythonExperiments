"""
https://leetcode.com/problems/partition-labels/

A string S of lowercase letters is given.
We want to partition this string into as many parts as possible so that each letter appears in at most one part,
and return a list of integers representing the size of these parts.
"""


from typing import List


class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        """
        This problem looks like the overlapping intervals
        - At each letter assign an interval of present [i,j]
        - Any letter that crosses belong to the same partition => join intervals
        - At the end, report the non joined intervals

        But we can do it in a more optimized manner:
        - identify the last occurrence of each letter from 'a' to 'z' (first scan)
        - scan from left to right to 'end_range' (initially 0)
        - for each letter, extend 'end_range' to its last occurrence
        - when you reach 'end_range', start a new partition
        """

        last_occurrence = {}
        for i, c in enumerate(s):
            last_occurrence[c] = i

        partitions = []
        start_range = 0
        end_range = 0
        for i, c in enumerate(s):
            end_range = max(end_range, last_occurrence[c])
            if i == end_range:
                partitions.append(end_range - start_range + 1)
                start_range = i + 1
                end_range = i + 1

        return partitions
