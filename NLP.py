import numpy as np

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
import unicodedata
import spacy


"""
General part of the pipeline
"""

class EraseTagButTense:
    def __init__(self):
        pass

    def fit(self, sentences, labels=None):
        pass

    def transform(self, sentences):
        for sentence in sentences:
            output = []
            for (token, tag) in sentence:
                if tag[0] == 'V':
                    output.append(token + '+' + tag)
                else:
                    output.append(token)
            yield output

"""
NLTK based pipeline for text pre-processing
"""

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('wordnet')

class NltkTextSplit:
    def __init__(self, stop_words=set(nltk.corpus.stopwords.words('english'))):
        self.stop_words = stop_words

    def fit(self, sentences, labels=None):
        return self

    def transform(self, sentences):
        return (self.normalize(sentence) for sentence in sentences)

    def normalize(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return [
            (word.lower(), tag)
            for (word, tag) in nltk.pos_tag(tokens)
            if not self.is_stopword(word) and not self.is_punctuation(word)
            ]

    def is_punctuation(self, token):
        return all(unicodedata.category(c).startswith('P') for c in token)

    def is_stopword(self, token):
        return token.lower() in self.stop_words


class NltkLemmatizer:
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()

    def fit(self, sentences, labels=None):
        return self

    def transform(self, sentences):
        return (self.normalize(sentence) for sentence in sentences)

    def normalize(self, sentence):
        return [(self.lemmatize(word, tag).lower(), tag) for (word, tag) in sentence]

    def lemmatize(self, word, tag):
        return self.lemmatizer.lemmatize(word, self.to_wordnet_tag(tag))

    def to_wordnet_tag(self, nltk_tag):
        if nltk_tag[0] == "V":
            return wn.VERB
        elif nltk_tag[0] == "J":
            return wn.ADJ
        elif nltk_tag[0] == "R":
            return wn.ADV
        return wn.NOUN

class NtlkStemmer:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def fit(self, sentences, labels=None):
        return self

    def transform(self, sentences):
        return (self.normalize(sentence) for sentence in sentences)

    def normalize(self, sentence):
        return [(self.stem(word).lower(), tag) for (word, tag) in sentence]

    def stem(self, word):
        return self.stemmer.stem(word)

def test_ntlk(text):
    print("--- NTLK Split ---")
    just_split = Pipeline(steps=[
        ('split', NltkTextSplit(stop_words={"the", "a"})),
        ('no_tags', EraseTagButTense())
    ])
    for x in just_split.transform(text):
        print(x)

    print("--- NTLK Stemming ---")
    wi_stem = Pipeline(steps=[
        ('split', NltkTextSplit(stop_words={"the", "a"})),
        ('stemmer', NtlkStemmer()),
        ('no_tags', EraseTagButTense())
    ])
    for x in wi_stem.transform(text):
        print(x)

    # Takes much more time !
    print("--- NTLK Lemmatizing ---")
    wi_lem = Pipeline(steps=[
        ('split', NltkTextSplit(stop_words={"the", "a"})),
        ('lemmatize', NltkLemmatizer()),
        ('no_tags', EraseTagButTense())
    ])
    for x in wi_lem.transform(text):
        print(x)

"""
Spacy based pipeline for text pre-processing
"""

# Requires to run: python3 -m spacy download en
# See excellent documentation: https://spacy.io/usage/models
# And the linguistic features: https://spacy.io/usage/linguistic-features

class SpacyProcessing:
    def __init__(self):
        self.tokenizer = spacy.load('en')

    def fit(self, sentences, labels=None):
        pass

    def transform(self, sentences):
        for sentence in sentences:
            yield [
                (token.lemma_, token.tag_)
                for token in self.tokenizer(sentence)
                if not self.is_punctuation(token.lemma_)
            ]

    def is_punctuation(self, token):
        return all(unicodedata.category(c).startswith('P') for c in token)

def test_spacy(text):
    print("--- Spacy ---")
    tf = Pipeline(steps=[
        ('spacy', SpacyProcessing()),
        ('no_tags', EraseTagButTense())
    ])
    for x in tf.transform(text):
        print(x)


"""
Running examples
"""

example_text = [
    "The goal is to try to use a natural language processing library such as NLTK.",
    "Let us see if I can succeed in that task as I succeeded before.",
    "Refactoring of the MEF class into C++ or Cpp class"]

test_ntlk(example_text)
test_spacy(example_text)
