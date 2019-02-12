"""
https://leetcode.com/problems/partition-labels/
"""

from typing import List


class Solution:
    def partitionLabels(self, S: str) -> List[int]:
        """
        This problem looks like the overlapping intervals
        - At each letter assign an interval of present [i,j]
        - Any letter that crosses belong to the same partition => join intervals
        - At the end, report the non joined intervals
        """
        # TODO
