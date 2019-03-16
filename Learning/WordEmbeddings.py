from annoy import AnnoyIndex

import fastText
import itertools
import numpy as np

import torch
import torch.nn as nn

from Learning.Vocabulary import *
from Learning.Tokenizer import *


def prepare_fast_text_unsupervised():
    corpus = CommitMessageCorpus.from_file('resources/perforce_cl_unsupervised.txt', keep_unclassified=True)
    tokenizer = NltkTokenizer()
    with open('resources/perforce_cl_unsupervised_cleaned.txt', mode='w') as cleaned:
        for line in itertools.chain(corpus.get_inputs(), corpus.get_unclassified()):
            tokens = tokenizer(line)
            cleaned.writelines([" ".join(tokens), '\n'])

    '''
    with open('resources/perforce_cl_unsupervised.txt', mode='r') as raw:
        with open('resources/perforce_cl_unsupervised_cleaned.txt', mode='w') as cleaned:
            for line in raw:
                tokens = tokenizer(line)
                cleaned.writelines([" ".join(tokens), '\n'])
    '''


class WordEmbeddings:
    def __init__(self, word_to_index, unsupervised_matrix):
        for reserved in Vocabulary.RESERVED:
            word_to_index[reserved] = word_to_index['</s>'] # TODO - HACK to map fastText unknown to my unknown
        self.vocabulary = Vocabulary(word_to_index)
        self.embedding_size = len(unsupervised_matrix[0])
        self.embedding = nn.Embedding(num_embeddings=len(self.vocabulary), embedding_dim=self.embedding_size)
        self.embedding.weight.data = torch.FloatTensor(unsupervised_matrix)
        self.embedding.weight.requires_grad = False

    def get_vocabulary(self) -> Vocabulary:
        return self.vocabulary

    def get_embedding(self) -> nn.Embedding:
        return self.embedding

    def get_vector(self, key):
        if isinstance(key, str):
            key = self.vocabulary.word_lookup(key)
        return self.embedding.weight.data[key] if key is not None else None

    def to_search_index(self) -> AnnoyIndex:
        """
        To do proximity / analogy queries
        """
        index = AnnoyIndex(self.embedding_size)
        for word in self.vocabulary.words():
            word_index = self.vocabulary.word_lookup(word)
            index.add_item(word_index, self.embedding.weight.data[word_index])
        index.build(5)
        return index

    def transfer_learning_to(self, vocabulary: Vocabulary, embedding_module: nn.Embedding):
        """
        Fill a foreign embedding module (with its own dictionary) with the pre trained embeddings
        """
        for word in vocabulary.words():
            word_index = vocabulary.word_lookup(word)
            vector = self.get_vector(word)
            if vector is not None:
                embedding_module.weight.data[word_index] = vector

    @staticmethod
    def learn_from(input_file, embedding_size, output_file):
        unsupervised_model = fastText.train_unsupervised(input=input_file, dim=embedding_size, model='skipgram')
        unsupervised_model.save_model(output_file)

    @classmethod
    def load_from(cls, model_path):
        unsupervised_model = fastText.load_model(model_path)
        unsupervised_matrix = np.stack([unsupervised_model.get_word_vector(w) for w in unsupervised_model.get_words()])
        word_to_index = {w: unsupervised_model.get_word_id(w) for w in unsupervised_model.get_words()}
        return cls(word_to_index, unsupervised_matrix)


def run_unsupervised_learning(embedding_size):
    WordEmbeddings.learn_from(input_file='resources/perforce_cl_unsupervised_cleaned.txt',
                              embedding_size=embedding_size,
                              output_file='resources/unsupervised_model.bin')


def test_unsupervised_learning():
    embeddings = WordEmbeddings.load_from(model_path='resources/unsupervised_model.bin')
    search_index = embeddings.to_search_index()

    while True:
        query = input("query>")
        word_vector = embeddings.get_vector(query)
        word_indices = search_index.get_nns_by_vector(word_vector, n=10)
        print([embeddings.get_vocabulary().index_lookup(word_index) for word_index in word_indices])


# TODO - seems to have some bugs in the Tokenizer...
# TODO - add stemming?
# prepare_fast_text_unsupervised()
# run_unsupervised_learning(embedding_size=20)

"""
query>refactor
['refactor', 'refactore', 'refacto', 'refactorize', 'centralize', 'refactoring', 'rework', 're-implement', 'refactored', 'rewrite']
query>clean
['clean', 'remove', 'cleaning', 'uncessary', 'obsolete', 'cleaning/refactoring', 'unused', 'deprecated', 'unncessary', 'code']
query>fix
['fix', 'non-regression', 'mismerge', 'correcting', 'issues', 'caused', 'typo', 'mainstream', 'uncovered', 'detecting']
query>correct
['correct', 'wrong', 'incoherence', 'corrects', 'mismatch', 'corrected', 'mismatched', 'convertion', 'instread', 'mlk']
query>improve
['improve', 'improves', 'improvement', 'performance', 'enhance', 'enhancing', 'performances', 'optimize', 'testability', 'optimizing']
query>move
['move', 'remove', 'cleanup/modernize', 'moved', 'renaming', 'reorganization', 'iba', 'deprecate', 'migrate', 'unused']
query>feature
['feature', 'mop', 'light', 'offer', 'redesign', 'terminology', 'pricing/cash', 'modality', 'use/design', 'ergonomic']
query>class
['class', 'acessors', 'classe', 'clas', 'lpp', 'mef', 'short-cut', 'mefclass', 'superclass', 'enums']
query>methodm
['method', 'methode', 'methods', 'constructors', 'parameterize', 'motor', 'methodes', 'smc', 'constructor', 'factory']
query>odule
['<unknown>', 'and', 'specific', 'new', 'fro', 'without', 'using', 'concerning', 'projecting', 'enriched']
query>module
['module', 'rate/credit', 'yield', 'securities/futures', 'credit', 'priceyield', 'price-yield', 'securities', 'antithetic', 'security']
query>simplify
['simplify', 'simplifying', 'refactore', 'rework', 'refactorize', 'refacto', 'rewrite', 'refactor', 'refactorisation', 'simplified']
"""

# test_unsupervised_learning()
