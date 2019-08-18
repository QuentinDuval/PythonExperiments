"""
https://leetcode.com/problems/maximum-product-of-word-lengths/

Given a string array words, find the maximum value of length(word[i]) * length(word[j]) where the two words
do not share common letters.

You may assume that each word will contain only lower case letters.
If no such two words exist, return 0.
"""

from collections import defaultdict
import string
from typing import List


class Solution:
    def maxProduct(self, words: List[str]) -> int:
        n = len(words)
        if n <= 1:
            return 0

        '''
        words.sort(key=len, reverse=True)
        masks = [self.mask_of(w) for w in words]
        max_len = len(words[0])

        max_prod = 0
        for i in range(n-1):
            if len(words[i]) * max_len <= max_prod:
                break
            for j in range(i+1, n):
                if masks[i] & masks[j] == 0:
                    max_prod = max(max_prod, len(words[i]) * len(words[j]))
                    break
                elif len(words[i]) * len(words[j]) <= max_prod:
                    break
        return max_prod
        '''

        sets_to_len = defaultdict(int)
        for w in words:
            mask = self.mask_of(w)
            sets_to_len[mask] = max(sets_to_len[mask], len(w))

        max_product = 0
        for s1, l1 in sets_to_len.items():
            for s2, l2 in sets_to_len.items():
                if not s1 & s2:
                    max_product = max(max_product, l1 * l2)
        return max_product

    def mask_of(self, w: str) -> int:
        mask = 0
        for c in string.ascii_lowercase:
            mask = mask << 1
            if c in w:
                mask += 1
        return mask
