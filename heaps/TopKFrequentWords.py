"""
https://leetcode.com/problems/top-k-frequent-words/

Given a non-empty list of words, return the k most frequent elements.

Your answer should be sorted by frequency from highest to lowest.
If two words have the same frequency, then the word with the lower alphabetical order comes first.
"""

import heapq
from typing import List


class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        """
        Avoid sorting the whole collection by using heaps
        TODO - Alternatively, could use a partial sort based on quick sort (partition but only deal with lower half)
        """
        counter = {}
        for word in words:
            counter[word] = counter.get(word, 0) + 1

        ranks = [(-count, word) for word, count in counter.items()]
        heapq.heapify(ranks)
        return [heapq.heappop(ranks)[1] for _ in range(k)]
