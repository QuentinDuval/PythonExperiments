"""
https://leetcode.com/problems/distant-barcodes/

In a warehouse, there is a row of barcodes, where the i-th barcode is barcodes[i].

Rearrange the barcodes so that no two adjacent barcodes are equal.
You may return any answer, and it is guaranteed an answer exists.
"""

import heapq
from typing import List


class Solution:
    def rearrangeBarcodes(self, barcodes: List[int]) -> List[int]:
        """
        Heap based solution:
        - pick the number with highest frequence and lowest value
        - except if it was the one put previously, look at the second one
        Complexity is O(N log N) beats 18% (624 ms)
        """

        counts = {}
        for code in barcodes:
            counts[code] = counts.get(code, 0) + 1

        heap = [(-count, code) for code, count in counts.items()]
        heapq.heapify(heap)

        def add_to_heap(count, code):
            if count:
                heapq.heappush(heap, (count, code))

        result = []
        while heap:
            count, code = heapq.heappop(heap)
            if not result or code != result[-1]:
                result.append(code)
                add_to_heap(count + 1, code)
            else:
                count2, code2 = heapq.heappop(heap)
                result.append(code2)
                add_to_heap(count2 + 1, code2)
                add_to_heap(count, code)
        return result
