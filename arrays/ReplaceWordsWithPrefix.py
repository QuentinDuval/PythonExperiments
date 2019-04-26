"""
GREAT PROBLEM

https://leetcode.com/problems/replace-words

In English, we have a concept called root, which can be followed by some other words to form another longer word - let's call this word successor. For example, the root an, followed by other, which can form another word another.

Now, given a dictionary consisting of many roots and a sentence. You need to replace all the successor in the sentence with the root forming it. If a successor has many roots can form it, replace it with the root with the shortest length.

You need to output the sentence after the replacement.
"""


from typing import List


class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        """
        Trie idea:
        - For each word in the sentence, search for the smallest prefix
        - Replace the word with the prefix if found, else keep it the same
        """
        dictionary.sort()
        dictionary = self.filter_smallest_prefixes(dictionary)
        cleaned = []
        for word in sentence.split(" "):
            prefix = self.search_prefix(dictionary, word)
            cleaned.append(prefix if prefix else word)
        return " ".join(cleaned)

    @staticmethod
    def filter_smallest_prefixes(dictionary):
        prev = None
        write = 0
        for read in range(len(dictionary)):
            if not prev or not dictionary[read].startswith(prev):
                dictionary[write] = dictionary[read]
                prev = dictionary[write]
                write += 1
        return dictionary[:write]

    @staticmethod
    def search_prefix(dictionary, word):
        """
        [bat, bot, cat, rat]
        - looking for 'category' - want to find 'cat'
        - looking for 'cottage' - want to find nothing
        'lo' will point to where to insert (next bigger word)
        'hi' will point to element just before (first smaller word)
        """
        lo = 0
        hi = len(dictionary) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if word == dictionary[mid]:
                return word
            elif word < dictionary[mid]:
                hi = mid - 1
            else:
                lo = mid + 1

        if hi >= 0 and word.startswith(dictionary[hi]):
            return dictionary[hi]
        return None
