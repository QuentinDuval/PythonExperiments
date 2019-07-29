"""
https://leetcode.com/problems/image-overlap/

Two images A and B are given, represented as binary, square matrices of the same size.
(A binary matrix has only 0s and 1s as values.)

We translate one image however we choose (sliding it left, right, up, or down any number of units), and place it on top
of the other image.  After, the overlap of this translation is the number of positions that have a 1 in both images.

(Note also that a translation does not include any kind of rotation.)

What is the largest possible overlap?
"""
from typing import List


class Solution:
    def largestOverlap(self, m1: List[List[int]], m2: List[List[int]]) -> int:
        """
        One solution is to try every (i, j) position in the first matrix and map it to second matrix, and check the overlap.
        - try all sliding of i between -H and H
        - try all sliding of j between -W and W
        => But it is too slow and timeouts

        IT PASSES BUT THIS IS WRONG:
        1) translate m1 for positive i and positive j slides
        2) translate m2 for positive i and positive j slides
        => We basically avoid negative i with positive j, and positive i with negative j

        Failing test case:
        [[0,0,0],[1,1,0],[0,0,0]]
        [[0,1,1],[0,0,0],[0,0,0]]

        To make the test case pass, you have to restrict the range size:
        for i in range(max(0, start_i), min(h, h + start_i)):
            for j in range(max(0, start_j), min(w, w + start_j)):
                if 1 == m1[i][j] == m2[i-start_i][j-start_j]:
                    count += 1
        """
        h = len(m1)
        w = h

        max_overlap = 0
        for i in range(-h, h):
            for j in range(-w, w):
                max_overlap = max(max_overlap, self.count_overlap(m1, m2, i, j))
        return max_overlap

    def count_overlap(self, m1, m2, start_i, start_j) -> int:
        h = len(m1)
        w = h

        count = 0
        for i in range(max(0, start_i), min(h, h + start_i)):
            for j in range(max(0, start_j), min(w, w + start_j)):
                if 1 == m1[i][j] == m2[i - start_i][j - start_j]:
                    count += 1
        return count
