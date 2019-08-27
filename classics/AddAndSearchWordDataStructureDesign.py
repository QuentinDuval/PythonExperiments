"""
https://leetcode.com/problems/add-and-search-word-data-structure-design/

Design a data structure that supports the following two operations:
* void addWord(word)
* bool search(word)

search(word) can search a literal word or a regular expression string containing only letters a-z or '.'.
A '.' means it can represent any one letter.
"""

class Trie:
    def __init__(self):
        self.has_val = False
        self.children = {}

    def add_word(self, word):
        node = self
        for c in word:
            child = node.children.get(c, None)
            if not child:
                child = Trie()
                node.children[c] = child
            node = child
        node.has_val = True

    def has_word(self, word):
        to_visit = [(self, 0)]
        while to_visit:
            node, pos = to_visit.pop()
            if pos == len(word):
                if node.has_val:
                    return True
                continue

            if word[pos] != '.':
                child = node.children.get(word[pos], None)
                if child is not None:
                    to_visit.append((child, pos + 1))
            else:
                for child in node.children.values():
                    to_visit.append((child, pos + 1))
        return False


class WordDictionary:

    def __init__(self):
        self.trie = Trie()

    def addWord(self, word: str) -> None:
        self.trie.add_word(word)

    def search(self, word: str) -> bool:
        return self.trie.has_word(word)

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)