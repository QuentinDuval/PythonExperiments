import string
import unicodedata
from typing import List

import nltk

from Learning.TokenParser import TokenParser


class Tokenizer:
    def __call__(self, sentence: str) -> List[str]:
        return self.tokenize(sentence)

    def tokenize(self, sentence: str) -> List[str]:
        pass


class SplitTokenizer(Tokenizer):
    def tokenize(self, sentence: str) -> List[str]:
        return [token for token in sentence.split(" ") if token not in string.punctuation]


class NltkTokenizer(Tokenizer):
    def __init__(self):
        self.token_parser = TokenParser()

    def tokenize(self, sentence: str) -> List[str]:
        tokens = []
        sentence = self.clean_sentence(sentence)
        for token in nltk.word_tokenize(sentence):
            if not self.is_punctuation(token):
                tokens.append(self.token_parser.parse(token))
        return tokens

    def clean_sentence(self, sentence: str) -> str:
        # TODO - replace "::" by "." for namespaces?
        return sentence.replace("&amp;", " and ")

    @staticmethod
    def is_punctuation(lemma: str) -> bool:
        if lemma == "\n":
            return True
        return all(unicodedata.category(c).startswith('P') for c in lemma)


class NGramTokenizer(Tokenizer):
    def __init__(self, tokenizer: Tokenizer, count=2, with_stop_words=True):
        self.tokenizer = tokenizer
        self.count = count
        if with_stop_words:
            self.stop_words = {"the", "in", "a"}    # TODO: enhance or even remove
        else:
            self.stop_words = set()

    def tokenize(self, sentence: str) -> List[str]:
        single_tokens = self.tokenizer.tokenize(sentence)
        tokens = list(single_tokens)

        single_tokens = [t for t in single_tokens if t not in self.stop_words]
        for shift in range(1, self.count):
            for i in range(shift, len(single_tokens)):
                tokens.append(" ".join(single_tokens[i-shift:i+1]))
        return tokens
