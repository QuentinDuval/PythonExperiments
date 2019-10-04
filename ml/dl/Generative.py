import torch.nn as nn

import nltk

# markov chain
# RNN

# nltk.word_tokenize()


class Writer(nn.Module):
    def __init__(self, embedding_size):
        self.embeddings = nn.Embedding(embedding_dim=embedding_size)
        # create a RNN that outputs a character

    def forward(self, before):
        # pass the characters through the embedding
        # pass through the RNN
        # output the best character
        pass


