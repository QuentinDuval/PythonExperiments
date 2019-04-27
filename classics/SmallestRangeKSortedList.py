from collections import defaultdict, Counter
from typing import List


# TODO - https://www.youtube.com/watch?v=zplklOy7ENo


class Solution:
    def smallestRange(self, lists: List[List[int]]) -> List[int]:
        """
        Idea:
        - merge the lists, keeping in memory where each elements are from (merge equal elements)
        - do a moving range (two pointers algorithm) to find the smallest range holding them all
        """

        merged = defaultdict(Counter)
        for i, l in enumerate(lists):
            for n in l:
                merged[n][i] += 1

        merged = list(sorted(merged.items(), key=lambda p: p[0]))

        best_lo, best_hi = 0, len(merged) - 1
        lo, hi = 0, -1
        found = Counter()  # It must be a counter (for there might be duplicates)
        while hi < len(merged) - 1:
            while len(found) < len(lists) and hi < len(merged) - 1:
                hi += 1
                found += merged[hi][1]

            while len(found) == len(lists):
                if merged[hi][0] - merged[lo][0] < merged[best_hi][0] - merged[best_lo][0]:
                    best_hi, best_lo = hi, lo
                found -= merged[lo][1]
                lo += 1

        return [merged[best_lo][0], merged[best_hi][0]]
