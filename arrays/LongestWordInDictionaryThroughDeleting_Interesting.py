"""
https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/

Given a string and a string dictionary, find the longest string in the dictionary that can be formed by deleting
some characters of the given string.

If there are more than one possible results, return the longest word with the smallest lexicographical order.
If there is no possible result, return the empty string.
"""

from typing import List


class Solution:
    def findLongestWord(self, s: str, d: List[str]) -> str:
        """
        Can do a kind of search in an sorted array for the dictionary
        (like for tries, do a lower bound search, check if the suffix match)

        This makes the search fast and allows us to try a lot of possibilities

        A trie would help to see the possibilities as well:
        - which letter should I select next?
        - then advance to that letter and check if the remaining length is enough...
        => this is a search oriented by the TRIE

        We could search directing by 's':
        - try to take a new letter: do we have matches? Yes, continue, else abort.
        => do not need a real TRIE here, just a sorted array

        But this is too slow, and much too complex:
        - we could just try every word in the dictionary
        - and check if this word is a subsequence of s
        => In the end, this is what we do if a we a trie !!! (we just avoid backing off when we do a TRIE)
        """

        if not d:
            return ""

        '''
        # Manual find takes 244 ms
        def is_subsequence(sub, full):
            pos = 0
            for i, c in enumerate(sub):
                while pos < len(full) and full[pos] != c:
                    pos += 1
                if len(full) - pos < len(sub) - i:  # Nice optim...
                    return False
                pos += 1
            return True

        d.sort(key=lambda w: (-len(w), w))
        for w in d:
            if is_subsequence(w, s):
                return w
        return ""
        '''

        # But built-in functions (without optim) takes 72 ms (98%)

        def is_subsequence(sub, full):
            pos = 0
            for i, c in enumerate(sub):
                pos = full.find(c, pos)
                if pos == -1:
                    return False
                pos += 1
            return True

        d.sort(key=lambda w: (-len(w), w))
        for w in d:
            if is_subsequence(w, s):
                return w
        return ""

        '''
        # TODO - failed attempt at using sorting and search in TRIE

        d.sort()
        min_word = min(len(w) for w in d)

        def search_prefix(prefix: str):
            lo = 0
            hi = len(d) - 1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if d[mid] < prefix:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return lo

        def select_best(longest: str, attempt: str) -> str:
            if len(attempt) > len(longest):
                return attempt
            elif len(attempt) == len(longest):
                return min(attempt, longest)
            return longest

        def search_longest(prev: str, pos: int) -> int:
            if len(s) - pos < min_word - len(prev):
                return ""

            if pos == len(s):
                return ""

            longest = ""
            for i in range(pos, len(s)):
                attempt = prev + s[i]
                where = search_prefix(attempt)
                if where < len(d):
                    if d[where] == attempt:
                        longest = select_best(longest, attempt)
                    if d[where].startswith(attempt):                        
                        found = search_longest(attempt, i+1)
                        longest = select_best(longest, found)
            return longest

        return search_longest("", 0)
        '''

