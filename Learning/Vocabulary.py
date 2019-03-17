from Learning.Corpus import CommitMessageCorpus
from Learning.Tokenizer import Tokenizer
from collections import Counter
from typing import *


class Vocabulary:
    PADDING = "<padding>"
    START = "<start>"
    END = "<end>"
    UNKNOWN = "<unknown>"
    RESERVED = [PADDING, START, END, UNKNOWN]

    def __init__(self, word_to_index: Dict[str, int]):
        self.word_to_index = word_to_index
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}

    def words(self):
        return self.word_to_index.keys()

    def word_lookup(self, word: str):
        index = self.word_to_index.get(word)
        if index is None:
            index = self.word_to_index.get(self.UNKNOWN)
        return index

    def index_lookup(self, i: int):
        return self.index_to_word[i]

    def __len__(self):
        return len(self.word_to_index)

    def __repr__(self):
        return repr(self.words())

    @classmethod
    def from_words(cls, words, add_unknowns=False):
        all_words = cls.RESERVED + list(words) if add_unknowns else words
        word_to_index = {all_words[i]: i for i in range(len(all_words))}
        return cls(word_to_index)

    @classmethod
    def from_corpus(cls, corpus: CommitMessageCorpus, tokenizer: Tokenizer, min_freq=5, add_unknowns=False):
        counts = Counter()
        for x, _ in corpus:
            for token in tokenizer(x):
                counts[token] += 1
        words = []
        for word, count in counts.items():
            if count >= min_freq:
                words.append(word)
        return cls.from_words(words, add_unknowns=add_unknowns)
