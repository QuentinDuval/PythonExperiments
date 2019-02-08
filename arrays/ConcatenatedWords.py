from functools import lru_cache
from typing import *


"""
https://leetcode.com/problems/concatenated-words
Given a list of words (without duplicates), please write a program that returns all concatenated words in the given list of words.
A concatenated word is defined as a string that is comprised entirely of at least two shorter words in the given array.
"""


def findAllConcatenatedWordsInADict_1(words: List[str]) -> List[str]:
    """
    Idea of a TRIE:
    - sort the words by increase sizes
    - test if each string is inside the trie, and the rest is also inside the trie recursively
    - only add a word in the trie if it is not the concatenation of other words

    To test if a word is a concatenation of word:
    - Depth First Search on the TRIE: get the list of valid next words, and recur
    """
    words.sort(key=lambda s: len(s))

    output = []
    trie = NodeTrie()
    for word in words:
        if is_concatenated(trie, word):
            output.append(word)
        else:
            trie.insert(word)
    return output


class NodeTrie:
    def __init__(self):
        self.marked = False
        self.children = {}

    def insert(self, s):
        if not s:
            self.marked = True
        else:
            child = self.children.get(s[0], None)
            if child is None:
                child = NodeTrie()
                self.children[s[0]] = child
            child.insert(s[1:])

    def suffixes(self, s):
        suffixes = []
        node = self
        for i in range(len(s)):
            node = node.children.get(s[i])
            if node is None:
                break
            if node.marked:
                suffixes.append(s[i + 1:])
        return suffixes


def is_concatenated(trie, word):
    if not word:
        return False

    discovered = {word}
    toVisit = [word]
    while toVisit:
        word = toVisit.pop()
        if word == "":
            return True

        for suffix in trie.suffixes(word):
            if suffix not in discovered:
                toVisit.append(suffix)
                discovered.add(suffix)
    return False


"""
GREAT IMPROVEMENT (same algorithm - but no need for a TRIE, only a set)
"""


def findAllConcatenatedWordsInADict(words: List[str]) -> List[str]:
    """
    Idea of a TRIE:
    - sort the words by increase sizes
    - test if each string is inside the trie, and the rest is also inside the trie recursively
    - only add a word in the trie if it is not the concatenation of other words

    To test if a word is a concatenation of word:
    Depth First Search on the TRIE: get the list of valid next words, and recur

    BUT then we can realize that the TRIE does not help much... just use a set of words
    """
    word_set = set(w for w in words if w != "")

    if not word_set:
        return []

    min_len = min(len(s) for s in word_set)

    @lru_cache(maxsize=None)
    def check(s) -> bool:
        for i in range(min_len, len(s) - min_len + 1):
            if s[:i] in word_set:
                if s[i:] in word_set or check(s[i:]):
                    return True
        return False

    return [s for s in word_set if check(s)]

