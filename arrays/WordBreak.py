from functools import lru_cache
from typing import List


"""
Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, add spaces in s to construct
a sentence where each word is a valid dictionary word. Return all such possible sentences.

Note:
- The same word in the dictionary may be reused multiple times in the segmentation.
- You may assume the dictionary does not contain duplicate words.

Input:
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]

Output:
[
  "cats and dog",
  "cat sand dog"
]
"""


def wordBreak(s: str, words: List[str]) -> List[str]:
    """
    Dynamic programming seems appropriate:
    - There is an obvious sub-problem structure: one problem for each index
    - These sub-problems overlap: two paths to 'dog' in the main example

    Now the problem is how do we explore the graph without trying all the words.
    - A TRIE data structure can help: we move though the TRIE and find all suffixes
    - Sorting the dictionary can help too: SUFFIX tree we binary search into:
    ["and", "cat", "cats", "dog", "sand"]
    > "catsanddog" will return 'dog' => go left to find prefixes 'cat' and 'cats'
    > "sanddog" will return the end => go left to find the prefix 'sand'
    > "dog" will return "dog" => found a match! report it

    But there is a much simpler way:
    - Just store the dictionary into a set
    - Search only for prefixes that have a length that in the range of those in the dictionary
    """
    if not s or not words:
        return []

    n = len(s)
    min_word_len = min(len(s) for s in words)
    max_word_len = max(len(s) for s in words)
    word_set = set(words)

    @lru_cache(maxsize=None)
    def backtrack(start_pos):
        solutions = []
        for end_pos in range(start_pos + min_word_len, min(start_pos + max_word_len, n) + 1):
            w = s[start_pos:end_pos]
            if w not in word_set:
                continue

            if end_pos == n:
                solutions.append(w)
            else:
                solutions.extend(w + " " + sol for sol in backtrack(end_pos))
        return solutions

    return backtrack(start_pos=0)
