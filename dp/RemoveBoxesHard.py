"""
https://leetcode.com/problems/remove-boxes

Given several boxes with different colors represented by different positive numbers.
You may experience several rounds to remove boxes until there is no box left.
Each time you can choose some continuous boxes with the same color (composed of k boxes, k >= 1), remove them and get k*k points.

Find the maximum points you can get.
"""


from functools import lru_cache
from typing import List


class Box:
    def __init__(self, val):
        self.val = val
        self.count = 1

    def __repr__(self):
        return repr((self.val, self.count))


class Solution:
    def removeBoxes(self, boxes: List[int]) -> int:
        if not boxes:
            return 0

        # Compress the input data
        zipped = [Box(boxes[0])]
        for val in boxes[1:]:
            if val == zipped[-1].val:
                zipped[-1].count += 1
            else:
                zipped.append(Box(val))

        # TODO - THIS IS HARD:
        # - you cannot just mutate the ''.count' of the 'zipped' variable (otherwise the memoization would be wrong)
        # - you have to memoize the bonus_count and not mutate the 'zipped' table

        @lru_cache(maxsize=None)
        def visit(lo: int, hi: int, lo_bonus_count: int) -> int:
            if lo >= hi:
                return 0

            curr = zipped[lo]
            best_score = (curr.count + lo_bonus_count) ** 2 + visit(lo + 1, hi, 0)

            for i in range(lo + 2, hi):
                box = zipped[i]
                if box.val == curr.val:
                    score = visit(lo + 1, i, 0) + visit(i, hi, lo_bonus_count + curr.count)
                    best_score = max(score, best_score)

            return best_score

        return visit(0, len(zipped), 0)