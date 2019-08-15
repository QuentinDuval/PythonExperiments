"""
https://leetcode.com/problems/minimum-area-rectangle/

Given a set of points in the xy-plane, determine the minimum area of a rectangle formed from these points,
with sides parallel to the x and y axes.

If there isn't any rectangle, return 0.
"""


from collections import defaultdict
from typing import List


class Solution:
    def minAreaRect(self, points: List[List[int]]) -> int:
        """
        A rectangle with sides parallel to X and Y axis will have a the following pattern:
        - x < y
        - (x,x), (x,y), (y,x), (y,y)
        WRONG... that would be true for a square.
        => CHECK YOUR HYPOTHESES

        A rectangle will have the pattern:
        (x0,y0), (x0,y1), (x1,y0), (x1,y1)

        First algorithm:
        - Sort the Xs and Ys by increasing order
        - Try all combinations of x0 < x1 and y0 < y1
        - Check if the pattern (x0,y0), (x0,y1), (x1,y0), (x1,y1) is found

        This is not particularly clever, maybe the (x0, y0) does not even exist!
        - Group the Ys by Xs
        - Try all pairs of Xs and intersect to find the Ys
        - Sort those Ys and try all combinations? No need, just find the minimum.
        - To make this simple, just sort the Ys when you group by Xs (just sort by Ys at the start)

        Complexity is O(X ^ 2 * Y)
        Beats 46%.
        """

        group_by_x = defaultdict(list)
        points.sort(key=lambda p: p[1])
        for x, y in points:
            group_by_x[x].append(y)

        min_rectangle = float('inf')
        xs = list(sorted(group_by_x.keys()))
        for i, x0 in enumerate(xs):
            for x1 in xs[i + 1:]:
                w = self.smallest_width_in_both(group_by_x[x0], group_by_x[x1])
                if y is not None:
                    min_rectangle = min(min_rectangle, w * (x1 - x0))
        return min_rectangle if min_rectangle != float('inf') else 0

    def smallest_width_in_both(self, ys1, ys2):
        min_width = float('inf')
        prev = None
        i1 = i2 = 0
        while i1 < len(ys1) and i2 < len(ys2):
            if ys1[i1] == ys2[i2]:
                if prev is not None:
                    min_width = min(min_width, ys1[i1] - prev)
                prev = ys1[i1]
                i1 += 1
                i2 += 1
            elif ys1[i1] < ys2[i2]:
                i1 += 1
            else:
                i2 += 1
        return min_width

