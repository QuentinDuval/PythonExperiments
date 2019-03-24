import unittest

from Learning.Tokenizer import *


class TokenizerTests(unittest.TestCase):

    def test_n_grams(self):
        tokenizer = NGramTokenizer(tokenizer=NltkTokenizer(), count=3, with_stop_words=True)
        res = set(tokenizer("this is a simple sentence"))
        self.assertSetEqual(res, {
            "this", "is", "a", "simple", "sentence",
            "this is", "is simple", "simple sentence",
            "this is simple", "is simple sentence"
        })
