"""
https://leetcode.com/problems/implement-magic-dictionary

Implement a magic directory with buildDict, and search methods.

For the method buildDict, you'll be given a list of non-repetitive words to build a dictionary.

For the method search, you'll be given a word, and judge whether if you modify *exactly* one character into another
character in this word, the modified word is in the dictionary you just built.
"""

from typing import List


class Origins:
    def __init__(self, origin):
        self.origin = origin
        self.multiple_elements = False

    def add(self, word):
        if word != self.origin:
            self.multiple_elements = True

    def accept(self, word):
        return self.multiple_elements or self.origin != word


class MagicDictionary:
    """
    Since the letters a-z are used, but no other, we can use a dummy character '.' to represent any "mutation".
    - When we add a word to the vocabulary, we add all its len(word) mutations in the dictionary
    - When we search a word, we look for all its mutation:
        - we check that either this mutation has 2 origins ("hello" and "helli" means that both are always accepted)
        - or that the origin is not the word itself (banish "hello" if it is the only origin of "h.llo")
    """

    def __init__(self):
        self.vocab = {}

    def buildDict(self, vocab: List[str]) -> None:
        for word in vocab:
            for mutation in self.mutations(word):
                origins = self.vocab.get(mutation, None)
                if origins is not None:
                    origins.add(word)
                else:
                    self.vocab[mutation] = Origins(word)

    def search(self, word: str) -> bool:
        for mutation in self.mutations(word):
            origins = self.vocab.get(mutation, None)
            if origins is not None and origins.accept(word):
                return True
        return False

    def mutations(self, word: str):
        for i in range(len(word)):
            yield word[:i] + '.' + word[i + 1:]

# Your MagicDictionary object will be instantiated and called as such:
# obj = MagicDictionary()
# obj.buildDict(dict)
# param_2 = obj.search(word)