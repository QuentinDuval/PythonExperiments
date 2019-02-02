import unittest

from backtrack.Exercises import *
from dictionaries.tries import NodeTrie


class TestBackTrack(unittest.TestCase):

    def test_derangement(self):
        expected = [[2, 3, 1], [3, 1, 2]]
        self.assertListEqual(expected, derangement(3))

    def test_multiset_permutation(self):
        expected = [[1, 1, 2, 2], [1, 2, 1, 2], [1, 2, 2, 1], [2, 1, 1, 2], [2, 1, 2, 1], [2, 2, 1, 1]]
        self.assertListEqual(expected, multiset_permutations([1, 1, 2, 2]))

    def test_keypad_words(self):
        dico = NodeTrie()
        dico.add("hello")
        self.assertListEqual(["hello"], keypad_words([4, 3, 5, 5, 6], dico))

