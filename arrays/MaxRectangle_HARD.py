"""
https://leetcode.com/problems/maximal-rectangle/

Given a 2D binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.
"""


from typing import List


class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        """
        Pre-treating the inputs so that we have the numbers of 1 above for each row
        Doing so let us go back to the 'max rectangle in histograph' problem
        """
        if not matrix or not matrix[0]:
            return 0

        h = len(matrix)
        w = len(matrix[0])

        ones_above = [[0] * w for _ in range(h)]
        for j in range(w):
            above = 0
            for i in range(h):
                if matrix[i][j] == '1':
                    above += 1
                else:
                    above = 0
                ones_above[i][j] = above

        return max(self.max_histogram(row) for row in ones_above)

    def max_histogram(self, heights: List[int]) -> int:
        """
        Accumulate the heights on a stack the following way:
        - when the height increases, add pair (index, new_height)
        - when the height decreases:
            - pop the heigher pair that are higher and compute area
            - add a new pair with (last_index_poped, height)
        - when the height is equal, do nothing

        Invariant:
        - the stack contains strictly increasing heights
        - so computing the area is easy (just look to right when popping)
        - the (index, height) indicates from which 'index' the height is higher than height
        """

        stack = [(-1, 0)]  # To be sure we do not pop all the stack
        max_area = 0
        heights.append(0)  # To be sure to pop the last rectangle
        for i, h in enumerate(heights):
            start = i
            while stack[-1][1] > h:
                start, prev_h = stack.pop()
                area = (i - start) * prev_h
                max_area = max(max_area, area)

            if stack[-1][1] < h:
                stack.append((start, h))
        return max_area
