from typing import List

import nltk
import string
import unicodedata


class NltkTokenizer:
    def __init__(self):
        pass

    def tokenize(self, sentence: str) -> List[str]:
        tokens = []
        sentence = self.clean_sentence(sentence)
        for token in nltk.word_tokenize(sentence):
            if not self.is_punctuation(token):
                tokens.append(token)
        return tokens

    def clean_sentence(self, sentence: str) -> str:
        return sentence.replace("&amp;", " and ").replace("::", ".")

    @staticmethod
    def is_punctuation(lemma: str) -> bool:
        if lemma == "\n":
            return True
        return all(unicodedata.category(c).startswith('P') for c in lemma)
