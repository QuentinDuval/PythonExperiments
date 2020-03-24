"""
https://leetcode.com/problems/longest-string-chain/

Given a list of words, each word consists of English lowercase letters.

Let's say word1 is a predecessor of word2 if and only if we can add exactly one letter anywhere in word1 to make it
equal to word2.  For example, "abc" is a predecessor of "abac".

A word chain is a sequence of words [word_1, word_2, ..., word_k] with k >= 1, where word_1 is a predecessor of word_2,
word_2 is a predecessor of word_3, and so on.

Return the longest possible length of a word chain with words chosen from the given list of words.
"""


from typing import List


class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        """
        - sort the strings by length O(N log N)
        - DO NOT transform each string into canonical form: order matters
        - create a graph of ancestor O(N ** 2)
        - look for the longest path in the graph O(N * avg degree) with memoization
        => We can combine these last two steps into one
        => 1056 ms
        """

        '''
        words.sort(key=len)

        def is_ancestor(w1, w2) -> bool:            
            i = diff = 0
            while i < len(w1):
                if diff > 1:
                    return False
                if w1[i] != w2[i+diff]:
                    diff += 1
                else:
                    i += 1
            return True

        N = len(words)
        longest = [1] * N
        for node in reversed(range(N)):
            for neigh in range(node+1, N):
                w1 = words[node]
                w2 = words[neigh]
                if len(w1) == len(w2):
                    continue
                elif len(w1) + 1 < len(w2):
                    break
                elif is_ancestor(w1, w2):
                    longest[node] = max(longest[node], 1 + longest[neigh])
        return max(longest)
        '''

        """
        We can make much better by look at neighbors differently in a hash map
        """

        words.sort(key=len)

        N = len(words)
        longest_from = {}
        for node in range(N):
            w = words[node]
            longest_from[w] = 1
            for i in range(len(w)):
                ancestor = w[:i] + w[i + 1:]
                longest_from[w] = max(longest_from[w], 1 + longest_from.get(ancestor, 0))
        return max(longest_from.values())

