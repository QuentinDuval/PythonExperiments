import unittest

from Learning.Tokenizer import *


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = NltkTokenizer()

    def test_namespace_tokenization(self):
        result = self.tokenizer("move ns::sub_ns::hello_world to lib/hello/world")
        self.assertEqual(['move', '<function>', 'to', '<path>'], result)

    def test_package_tokenization(self):
        result = self.tokenizer("move pack.package.ClassName to lib/hello/world")
        self.assertEqual(['move', '<function>', 'to', '<path>'], result)

    def test_class_tokenization(self):
        # TODO - identify class names
        result = self.tokenizer("move pack.sub-pack.ClassName to lib/hello/world")
        self.assertEqual(['move', '<function>', 'to', '<path>'], result)

    def test_n_grams(self):
        tokenizer = NGramTokenizer(tokenizer=NltkTokenizer(), count=3, with_stop_words=True)
        res = set(tokenizer("this is a simple sentence"))
        self.assertSetEqual(res, {
            "this", "is", "a", "simple", "sentence",
            "this is", "is simple", "simple sentence",
            "this is simple", "is simple sentence"
        })
