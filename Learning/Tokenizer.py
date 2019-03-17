import nltk
import re
import string
from typing import List
import unicodedata


class Tokenizer:
    def __call__(self, sentence: str) -> List[str]:
        return self.tokenize(sentence)

    def tokenize(self, sentence: str) -> List[str]:
        pass


class SplitTokenizer(Tokenizer):
    def tokenize(self, sentence: str) -> List[str]:
        return [token for token in sentence.split(" ") if token not in string.punctuation]


class NltkTokenizer(Tokenizer):
    ISSUE_TAG = "<issue>"
    PATH_TAG = "<path>"
    ENTITY_NAME = "<entity>"
    FUNCTION_TAG = "<function>"
    NUMBER_TAG = "<number>"
    LANGUAGE_TAG = "<language>"

    def __init__(self):
        self.issue = re.compile("^[a-zA-Z]+[\-]?[0-9]+$")
        self.abbreviations = {"url", "ci", "raii", "bau", "slo", "api", "stl"}
        self.languages = {"c++", "mef", "java", "c", "cpp", "ant", "groovy", "js", "scala"}

    def tokenize(self, sentence: str) -> List[str]:
        tokens = []
        sentence = self.clean_sentence(sentence)
        for token in nltk.word_tokenize(sentence):
            if not self.is_punctuation(token):
                tokens.append(self.clean_token(token))
        return tokens

    def clean_sentence(self, sentence: str) -> str:
        return sentence.replace("&amp;", " and ")

    def clean_token(self, token: str) -> str:
        if token.lower() in self.abbreviations:
            return token

        if token.lower() in self.languages:
            return self.LANGUAGE_TAG

        if self.issue.match(token):
            return self.ISSUE_TAG

        try:
            int(token)
            return self.NUMBER_TAG
        except ValueError:
            pass

        if token.isupper():
            return self.ENTITY_NAME

        if "_" in token:
            return self.ENTITY_NAME if token.isupper() else self.FUNCTION_TAG

        if self.count(token, lambda c: c == "/") >= 2:
            return self.PATH_TAG

        # TODO - make function name more... exact
        if self.count(token[1:], lambda c: c.isupper()) >= 1:
            return self.FUNCTION_TAG

        return token.lower()

    @staticmethod
    def count(token, pred):
        return sum(1 if pred(c) else 0 for c in token)

    @staticmethod
    def is_punctuation(lemma: str) -> bool:
        if lemma == "\n":
            return True
        return all(unicodedata.category(c).startswith('P') for c in lemma)


class BiGramTokenizer(Tokenizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.stop_words = {"the", "in", "a"} # TODO - stop words at the level above? Or compose it :)

    def tokenize(self, sentence: str) -> List[str]:
        single_tokens = self.tokenizer.tokenize(sentence)
        single_tokens = [t for t in single_tokens if t not in self.stop_words]
        tokens = [single_tokens[0]]
        for i in range(1, len(single_tokens)):
            tokens.append(single_tokens[i-1] + " " + single_tokens[i])
            tokens.append(single_tokens[i])
        return tokens
