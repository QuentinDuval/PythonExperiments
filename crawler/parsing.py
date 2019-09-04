from typing import List

import os
import nltk
import string
import unicodedata


stop_words = {
    "the", "of", "a", "to", "is", "am", "and", "in", "that", "this", "it", "we", "you"
}


def list_files_in_folder(folder: str):
    files = []
    for dir_path, dir_names, file_names in os.walk(folder):
        files.extend(os.path.join(folder, file_name) for file_name in file_names)
    return files


class NltkTokenizer:
    def __init__(self):
        pass

    def tokenize(self, sentence: str) -> List[str]:
        tokens = []
        sentence = self.clean_sentence(sentence)
        for token in nltk.word_tokenize(sentence):
            if not self.is_punctuation(token):
                tokens.append(token.lower())
        return tokens

    def clean_sentence(self, sentence: str) -> str:
        return sentence.replace("&amp;", " and ").replace("::", ".")

    @staticmethod
    def is_punctuation(lemma: str) -> bool:
        if lemma == "\n":
            return True
        return all(unicodedata.category(c).startswith('P') for c in lemma)
